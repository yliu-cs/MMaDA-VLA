import os
import json
import torch
import autoroot
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from typing import List, Dict, Union
from mmadavla.utils.prompt import Prompting
from mmadavla.models.mmada import MMaDAConfig
from argparse import ArgumentParser, Namespace
from accelerate.utils import DistributedType, set_seed
from transformers import get_cosine_schedule_with_warmup
from mmadavla.models.action_tokenizer import ActionTokenizer
from mmadavla.models.mmadavla import MMaDAVLAConfig, MMaDAVLAModelLM
from mmadavla.utils.misc import quiet, str_datetime, count_params, hash_str
from mmadavla.utils.diffusion import cosine_mask_schedule, mask_or_random_replace_tokens


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pretrained_mmada", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "MMaDA-8B-Base"))
    parser.add_argument("--pretrained_mmadavla", type=str, default=None)
    parser.add_argument("--action_chunk_size", type=int, default=5)
    parser.add_argument("--data_paths", nargs='+', type=str, default=list(glob(os.path.join(os.sep, "liuyang", "Dataset", "MMaDA-VLA", "TBD", "*.parquet"))))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.getcwd(), ".cache_dir"))
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA"))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--batch_size_per_gpu", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--warmup_ratio", type=int, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--mask_contiguous_region_prob", type=float, default=None)
    args = parser.parse_args()
    return args


def load_pretrained_models(args: Namespace) -> MMaDAVLAModelLM:
    if args.pretrained_mmadavla is None:
        action_tokenizer = ActionTokenizer(action_chunk_size=args.action_chunk_size, action_dim=7)
        base_config = MMaDAConfig.from_pretrained(args.pretrained_mmada).to_dict()
        base_config.update({
            "architectures": ["MMaDAVLAModelLM"],
            "new_vocab_size": base_config["new_vocab_size"] + action_tokenizer.vocab_size,
            "vision_codebook_size": base_config["codebook_size"],
            "vision_num_vq_tokens": base_config["num_vq_tokens"],
            "action_codebook_size": action_tokenizer.vocab_size,
            "action_num_vq_tokens": action_tokenizer.action_chunk_size * action_tokenizer.action_dim
        })
        del base_config["codebook_size"], base_config["num_vq_tokens"], base_config["auto_map"]
        mmadavla_config = MMaDAVLAConfig(**base_config)
        mmadavla_model = MMaDAVLAModelLM.from_pretrained(args.pretrained_mmada, torch_dtype=torch.bfloat16, config=mmadavla_config)
        mmadavla_model.resize_token_embeddings(mmadavla_model.config.new_vocab_size)
        mmadavla_model.config.embedding_size = mmadavla_model.config.vocab_size
    else:
        mmadavla_model = MMaDAVLAModelLM.from_pretrained(args.pretrained_mmadavla, torch_dtype=torch.bfloat16)
        if mmadavla_model.config.action_num_vq_tokens != args.action_chunk_size * 7:
            mmadavla_model.config.action_num_vq_tokens = args.action_chunk_size * 7
    return mmadavla_model


def save_checkpoint(
    save_dir: str,
    unwrap_mmadavla_model: MMaDAVLAModelLM,
    total_losses: List[float],
    mask_rates: List[float],
    args: Namespace,
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "total_losses.npy"), np.array(total_losses))
    np.save(os.path.join(save_dir, "mask_rates.npy"), np.array(mask_rates))
    with open(os.path.join(save_dir, "args.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)
    unwrap_mmadavla_model.save_pretrained(save_dir, safe_serialization=True)


def main(args: Namespace) -> None:
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        split_batches=True
    )
    if accelerator.is_main_process and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.print(f"{str_datetime()} Loading Models ...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_mmada, padding_side="left")
    mmadavla_model = load_pretrained_models(args)
    mmadavla_model = mmadavla_model.to(accelerator.device)
    accelerator.print(f"{str_datetime()} {'MMaDA-VLA Model Parameters':<30}: {count_params(mmadavla_model)}")
    mask_id = mmadavla_model.config.mask_token_id

    prompt = Prompting(
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        mask_id=mask_id,
        vision_codebook_size=mmadavla_model.config.vision_codebook_size,
        action_codebook_size=mmadavla_model.config.action_codebook_size
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config.update({"train_micro_batch_size_per_gpu": args.batch_size_per_gpu})
    if args.seed is not None:
        set_seed(args.seed)
    
    accelerator.print(f"{str_datetime()} Preparing Optimizer, DataLoader, Scheduler ...")
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in mmadavla_model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in mmadavla_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.beta[0], args.beta[1]),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    if accelerator.is_main_process and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
    train_dataset = load_dataset("parquet", data_files=args.data_paths, split="train", cache_dir=args.cache_dir)
    def collate_fn(batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, Union[List[str], torch.Tensor]]:
        task_inst = [f"{item['task_inst']}\n{item['robot_states']}" for item in batch]
        cur_image, pred_image, action = [torch.stack([torch.LongTensor(item[key]) for item in batch]) for key in ("cur_rgb", "goal_rgb", "action")]
        return {
            "task_inst": task_inst,
            "cur_image": cur_image,
            "pred_image": pred_image,
            "action": action
        }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        collate_fn=collate_fn,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int((len(train_dataloader) * args.num_train_epochs) * args.warmup_ratio),
        num_training_steps=(len(train_dataloader) * args.num_train_epochs)
    )
    mmadavla_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(mmadavla_model, optimizer, train_dataloader, lr_scheduler)
    mask_schedule = cosine_mask_schedule

    accelerator.print(f"{str_datetime()} Start Training ...")
    total_losses, mask_rates = [], []
    for epoch in range(args.num_train_epochs):
        mmadavla_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{args.num_train_epochs}]", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            task_inst, cur_image_tokens, pred_image_tokens, action_tokens = batch["task_inst"], batch["cur_image"], batch["pred_image"], batch["action"]
            cur_image_tokens, pred_image_tokens = [(key + len(prompt.tokenizer)) for key in (cur_image_tokens, pred_image_tokens)]
            action_tokens += len(prompt.tokenizer) + prompt.vision_codebook_size
            (pred_image_tokens, pred_image_labels), (action_tokens, action_labels), mask_prob = mask_or_random_replace_tokens(
                image_tokens=pred_image_tokens,
                action_tokens=action_tokens,
                mask_id=mask_id,
                args=args,
                mask_schedule=mask_schedule,
                is_training=True
            )
            input_ids, attn_bias, labels = prompt(
                task_inst=task_inst,
                cur_image_tokens=cur_image_tokens,
                pred_image_tokens=pred_image_tokens,
                action_tokens=action_tokens,
                pred_image_labels=pred_image_labels,
                action_labels=action_labels
            )
            loss = mmadavla_model.forward_process(
                input_ids=input_ids,
                attention_bias=attn_bias,
                labels=labels
            )
            accelerator.backward(loss)
            if args.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(mmadavla_model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_losses.append(accelerator.gather(loss.repeat(args.batch_size_per_gpu)).mean().item())
            mask_rates.append(accelerator.gather(mask_prob.repeat(args.batch_size_per_gpu)).mean().item())
            progress_bar.set_description(f"{str_datetime()} [Epoch {epoch + 1}/{args.num_train_epochs}] | Loss: {total_losses[-1]:.4f} | Masking Rate: {mask_rates[-1]:.4f}")
        if accelerator.is_main_process and (epoch + 1) % args.save_epoch == 0 and epoch != args.num_train_epochs - 1:
            accelerator.print(f"{str_datetime()} Saving Checkpoint into {os.path.join(args.output_dir, f'checkpoint_{epoch + 1}')} ...")
            save_checkpoint(
                save_dir=os.path.join(args.output_dir, f"checkpoint_{epoch + 1}"),
                unwrap_mmadavla_model=accelerator.unwrap_model(mmadavla_model),
                total_losses=total_losses,
                mask_rates=mask_rates,
                args=args
            )
    
    accelerator.wait_for_everyone()
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    if accelerator.is_main_process:
        save_checkpoint(
            save_dir=args.output_dir,
            unwrap_mmadavla_model=accelerator.unwrap_model(mmadavla_model),
            total_losses=total_losses,
            mask_rates=mask_rates,
            args=args
        )
    accelerator.print(f"{str_datetime()} Training End .")
    accelerator.end_training()


if __name__ == "__main__":
    quiet()
    args = get_args()
    args.data_paths = list(map(lambda x: x.replace("TBD", f"vla_{args.action_chunk_size}chunk"), args.data_paths))
    # args.output_dir = os.path.join(args.output_dir, hash_str(f"{args}{str_datetime()}"))
    main(args)
