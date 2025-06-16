import os
import math
import json
import torch
import autoroot
import numpy as np
from tqdm.auto import tqdm
from typing import List, Tuple
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer
from rold.utils.prompt import Prompting
from rold.models.magvitv2 import MagViTv2
from rold.models.mmada import MMaDAConfig
from rold.data.calvin import CalvinDataset
from rold.models.actrvq import ActionRVQModel
from argparse import ArgumentParser, Namespace
from rold.models.rold import RoLDConfig, RoLDModelLM
from accelerate.utils import DistributedType, set_seed
from transformers import get_cosine_schedule_with_warmup
from rold.utils.misc import quiet, str_datetime, count_params, hash_str
from rold.utils.diffusion import cosine_mask_schedule, mask_or_random_replace_tokens


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--pretrained_mmada", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "MMaDA-8B-Base"))
    parser.add_argument("--actrvq", type=str, default="36a391f3d2e0d405d7d39f100571a139")
    parser.add_argument("--task", type=str, default="ABC_D")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "ckpt", "RoLD"))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--batch_size_per_gpu", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--warmup_ratio", type=int, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--mask_contiguous_region_prob", type=float, default=None)
    args = parser.parse_args()
    return args


def load_pretrained_models(args: Namespace) -> Tuple[MagViTv2, ActionRVQModel, RoLDModelLM]:
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)
    pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.task.lower()}_{args.action_chunk_size}steps", args.actrvq)
    action_vq_model = ActionRVQModel.from_pretrained(pretrained_actrvq)
    action_vq_model.eval()
    action_vq_model.requires_grad_(False)
    base_config = MMaDAConfig.from_pretrained(args.pretrained_mmada).to_dict()
    base_config.update({
        "architectures": ["RoLDModelLM"],
        "new_vocab_size": base_config["new_vocab_size"] + action_vq_model.config.codebook_size,
        "vision_codebook_size": base_config["codebook_size"],
        "vision_num_vq_tokens": base_config["num_vq_tokens"],
        "action_codebook_size": action_vq_model.config.codebook_size,
        "action_num_vq_tokens": args.action_chunk_size * action_vq_model.config.num_quantizers
    })
    del base_config["codebook_size"], base_config["num_vq_tokens"], base_config["auto_map"]
    rold_config = RoLDConfig(**base_config)
    rold_model = RoLDModelLM.from_pretrained(args.pretrained_mmada, torch_dtype=torch.bfloat16, config=rold_config)
    rold_model.resize_token_embeddings(rold_model.config.new_vocab_size)
    rold_model.config.embedding_size = rold_model.config.vocab_size
    return vision_vq_model, action_vq_model, rold_model


def save_checkpoint(
    save_dir: str,
    unwrap_rold_model: RoLDModelLM,
    vision_losses: List[float],
    action_losses: List[float],
    total_losses: List[float],
    mask_rates: List[float],
    args: Namespace
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "vision_losses.npy"), np.array(vision_losses))
    np.save(os.path.join(save_dir, "action_losses.npy"), np.array(action_losses))
    np.save(os.path.join(save_dir, "total_losses.npy"), np.array(total_losses))
    np.save(os.path.join(save_dir, "mask_rates.npy"), np.array(mask_rates))
    with open(os.path.join(save_dir, "args.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)
    unwrap_rold_model.save_pretrained(save_dir, safe_serialization=True)


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
    vision_vq_model, action_vq_model, rold_model = load_pretrained_models(args)
    vision_vq_model, action_vq_model = vision_vq_model.to(accelerator.device), action_vq_model.to(accelerator.device)
    rold_model = rold_model.to(accelerator.device)
    accelerator.print(f"{str_datetime()} {'RoLD Model Parameters':<30}: {count_params(rold_model)}")
    mask_id = rold_model.config.mask_token_id

    prompt = Prompting(
        tokenizer=tokenizer,
        max_text_len=args.max_text_len,
        vision_codebook_size=rold_model.config.vision_codebook_size,
        action_codebook_size=rold_model.config.action_codebook_size
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config.update({"train_micro_batch_size_per_gpu": args.batch_size_per_gpu})
    if args.seed is not None:
        set_seed(args.seed)
    
    accelerator.print(f"{str_datetime()} Preparing Optimizer, DataLoader, Scheduler ...")
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in rold_model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in rold_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.beta[0], args.beta[1]),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    train_dataset = CalvinDataset(
        data_path=os.path.join(args.data_path, "training", f"calvin_{args.task.lower()}_{args.action_chunk_size}steps.npy"),
        image_path=os.path.join(args.data_path, "training", f"calvin_{args.task.lower()}_{args.action_chunk_size}steps_image.npy"),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int((len(train_dataloader) * args.num_train_epochs) * args.warmup_ratio),
        num_training_steps=(len(train_dataloader) * args.num_train_epochs)
    )
    rold_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(rold_model, optimizer, train_dataloader, lr_scheduler)
    mask_schedule = cosine_mask_schedule

    accelerator.print(f"{str_datetime()} Start Training ...")
    vision_losses, action_losses, total_losses, mask_rates, mses = [], [], [], [], []
    for epoch in range(args.num_train_epochs):
        rold_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{args.num_train_epochs}]", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            task_inst, cur_image, pred_image, action = batch["task_inst"], batch["cur_image"], batch["pred_image"], batch["action"]
            cur_image_tokens = vision_vq_model.get_code(cur_image.to(accelerator.device))
            pred_image_tokens = vision_vq_model.get_code(pred_image.to(accelerator.device))
            action_tokens = action_vq_model.tokenize(action.to(accelerator.device))
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
            input_ids, attn_mask, labels = prompt(
                task_inst=task_inst,
                cur_image_tokens=cur_image_tokens,
                pred_image_tokens=pred_image_tokens,
                action_tokens=action_tokens,
                pred_image_labels=pred_image_labels,
                action_labels=action_labels
            )
            (vision_logits, vision_loss), (action_logits, action_loss), loss = rold_model.forward_process(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels
            )
            with torch.no_grad():
                pred_action_ids = torch.argmax(action_logits.softmax(dim=-1), dim=-1) - (len(prompt.tokenizer) + prompt.vision_codebook_size)
                pred_action_ids = torch.clamp(pred_action_ids, max=prompt.action_codebook_size - 1, min=0)
                pred_action = action_vq_model.detokenize(pred_action_ids)
                mse = F.mse_loss(pred_action, action)
            accelerator.backward(loss)
            if args.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(rold_model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            vision_losses.append(accelerator.gather(vision_loss.repeat(args.batch_size_per_gpu)).mean().item())
            action_losses.append(accelerator.gather(action_loss.repeat(args.batch_size_per_gpu)).mean().item())
            total_losses.append(accelerator.gather(loss.repeat(args.batch_size_per_gpu)).mean().item())
            mask_rates.append(accelerator.gather(mask_prob.repeat(args.batch_size_per_gpu)).mean().item())
            mses.append(accelerator.gather(mse.repeat(args.batch_size_per_gpu)).mean().item())
            progress_bar.set_description(
                " | ".join([
                    f"{str_datetime()} [Epoch {epoch + 1}/{args.num_train_epochs}]",
                    f"Vision Loss: {vision_losses[-1]:.4f}",
                    f"Action Loss: {action_losses[-1]:.4f}",
                    f"Total Loss: {total_losses[-1]:.4f}",
                    f"Masking Rate: {mask_rates[-1]:.4f}",
                    f"MSE: {mses[-1]:.4f}"
                ])
            )
        accelerator.print(f"{str_datetime()} Saving Checkpoint into {os.path.join(args.output_dir, f'checkpoint_{epoch + 1}')} ...")
        if accelerator.is_main_process and epoch != args.num_train_epochs - 1:
            save_checkpoint(
                save_dir=os.path.join(args.output_dir, f"checkpoint_{epoch + 1}"),
                unwrap_rold_model=accelerator.unwrap_model(rold_model),
                vision_losses=vision_losses,
                action_losses=action_losses,
                total_losses=total_losses,
                mask_rates=mask_rates,
                args=args
            )
    
    accelerator.wait_for_everyone()
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    if accelerator.is_main_process:
        save_checkpoint(
            save_dir=args.output_dir,
            unwrap_rold_model=accelerator.unwrap_model(rold_model),
            vision_losses=vision_losses,
            action_losses=action_losses,
            total_losses=total_losses,
            mask_rates=mask_rates,
            args=args
        )
    accelerator.print(f"{str_datetime()} Training End .")
    accelerator.end_training()


if __name__ == "__main__":
    quiet()
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, hash_str(f"{args}{str_datetime()}"))
    main(args)