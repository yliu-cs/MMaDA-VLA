import os
import math
import json
import torch
import autoroot
from typing import Tuple
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer
from rold.models.magvitv2 import MagViTv2
from rold.models.mmada import MMaDAConfig
from rold.data.calvin import CalvinDataset
from rold.models.actrvq import ActionRVQModel
from argparse import ArgumentParser, Namespace
from rold.models.rold import RoLDConfig, RoLDModelLM
from accelerate.utils import DistributedType, set_seed
from transformers import get_cosine_schedule_with_warmup
from rold.utils.misc import quiet, str_datetime, count_params, hash_str


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pretrained_actrvq", type=str, default=os.path.join(os.getcwd(), "ckpt", "ActionRVQ"))
    parser.add_argument("--pretrained_visvq", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "magvitv2"))
    parser.add_argument("--pretrained_mmada", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "MMaDA-8B-Base"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--task", type=str, default="ABC_D")
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
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
    args = parser.parse_args()
    return args


def load_pretrained_models(args: Namespace) -> Tuple[MagViTv2, ActionRVQModel, RoLDModelLM]:
    vision_vq_model = MagViTv2.from_pretrained(args.pretrained_visvq)
    vision_vq_model.eval()
    vision_vq_model.requires_grad_(False)
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
    uni_prompting = None  # TODO: add uni-prompting
    vision_vq_model, action_vq_model, rold_model = load_pretrained_models(args)
    vision_vq_model, action_vq_model = vision_vq_model.to(accelerator.device), action_vq_model.to(accelerator.device)
    rold_model = rold_model.to(accelerator.device)
    accelerator.print(f"{str_datetime()} {count_params(vision_vq_model)=}")
    accelerator.print(f"{str_datetime()} {count_params(action_vq_model)=}")
    accelerator.print(f"{str_datetime()} {rold_model.config=}")
    accelerator.print(f"{str_datetime()} {count_params(rold_model)=}")
    mask_id = rold_model.config.mask_token_id

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
    mask_dtype = rold_model.get_input_embeddings().weight.dtype

    accelerator.print(f"{str_datetime()} Start Training ...")
    num_train_epochs, global_step = math.ceil(args.max_train_steps / (math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))), 0
    for epoch in range(num_train_epochs):
        rold_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{num_train_epochs}]", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(rold_model):
                loss, logits = rold_model.forward_process()  # TODO: complete forward process
                avg_loss = accelerator.gather(loss.repeat(args.batch_size_per_gpu)).mean()
                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(rold_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
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