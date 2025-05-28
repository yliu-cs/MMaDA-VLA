import os
import math
import json
import torch
import autoroot
from typing import Tuple
from tqdm.auto import tqdm
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
    parser.add_argument("--pretrained_actrvq", type=str, default=os.path.join(os.getcwd(), "ckpt", "ActionRVQ"))
    parser.add_argument("--pretrained_mmada", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "MMaDA-8B-Base"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--task", type=str, default="ABC_D")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "ckpt"))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--num_warmup_steps", type=int, default=5000)
    parser.add_argument("--max_train_steps", type=int, default=500000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--mask_contiguous_region_prob", type=float, default=None)
    args = parser.parse_args()
    return args


def load_pretrained_models(args: Namespace) -> Tuple[MagViTv2, ActionRVQModel, RoLDModelLM]:
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)
    args.pretrained_actrvq = os.path.join(os.getcwd(), "ckpt", f"ActRVQ_{args.task.lower()}_{args.action_chunk_size}steps")
    action_vq_model = ActionRVQModel.from_pretrained(args.pretrained_actrvq)
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


def main(args: Namespace) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        split_batches=True
    )

    accelerator.print(f"{str_datetime()} Loading Models ...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_mmada, padding_side="left")
    vision_vq_model, action_vq_model, rold_model = load_pretrained_models(args)
    vision_vq_model, action_vq_model = vision_vq_model.to(accelerator.device), action_vq_model.to(accelerator.device)
    rold_model = rold_model.to(accelerator.device)
    accelerator.print(f"{str_datetime()} {'Vision VQ Model Parameters':<30}: {count_params(vision_vq_model)}")
    accelerator.print(f"{str_datetime()} {'Action RVQ Model Parameters':<30}: {count_params(action_vq_model)}")
    accelerator.print(f"{str_datetime()} {'RoLD Model Parameters':<30}: {count_params(rold_model)}")
    mask_id = rold_model.config.mask_token_id

    prompt = Prompting(tokenizer=tokenizer, max_text_len=args.max_text_len)

    total_batch_size = args.batch_size_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps
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
        data_dir=os.path.join(args.data_dir, f"task_{args.task.upper()}", "training"),
        data_path=os.path.join(args.data_path, f"calvin_{args.task.lower()}_{args.action_chunk_size}steps.npy")
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=train_dataset.collate_fn)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)
    rold_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(rold_model, optimizer, train_dataloader, lr_scheduler)
    mask_schedule = cosine_mask_schedule

    accelerator.print(f"{str_datetime()} Start Training ...")
    num_train_epochs, global_step, losses, mask_rates = math.ceil(args.max_train_steps / (math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))), 0, [], []
    for epoch in range(num_train_epochs):
        rold_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{num_train_epochs}]", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            task_inst, cur_image, pred_image, action = batch["task_inst"], batch["cur_image"], batch["pred_image"], batch["action"]
            cur_image_tokens = vision_vq_model.get_code(cur_image.to(vision_vq_model.device))
            pred_image_tokens = vision_vq_model.get_code(pred_image.to(vision_vq_model.device))
            action_tokens = action_vq_model.tokenize(action.to(action_vq_model.device))
            cur_image_tokens, pred_image_tokens = [(key + len(prompt.tokenizer)) for key in (cur_image_tokens, pred_image_tokens)]
            action_tokens += len(prompt.tokenizer) + rold_model.config.vision_codebook_size
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
            with accelerator.accumulate(rold_model):
                logits, loss = rold_model.forward_process(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels
                )
                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(rold_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(accelerator.gather(loss.repeat(args.batch_size_per_gpu)).mean().item())
                mask_rates.append(accelerator.gather(mask_prob.repeat(args.batch_size_per_gpu)).mean().item())
                progress_bar.set_description(f"{str_datetime()} [Step {global_step + 1}/{args.max_train_steps}] | Loss: {losses[-1]:.4f} | Masking Rate: {mask_rates[-1]:.4f}")
            if accelerator.sync_gradients:
                global_step += 1
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
        (accelerator.unwrap_model(rold_model)).save_pretrained(args.output_dir, safe_serialization=True)
    accelerator.print(f"{str_datetime()} Training End .")
    accelerator.end_training()


if __name__ == "__main__":
    quiet()
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, hash_str(str(args)))
    main(args)