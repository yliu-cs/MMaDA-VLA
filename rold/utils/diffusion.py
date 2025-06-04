import math
import torch
import random
from argparse import Namespace
from typing import Tuple, Callable
from rold.utils.prompt import ignore_id


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def mask_by_random_topk(
    mask_len: torch.Tensor,
    probs: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator = None
) -> torch.Tensor:
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


def cosine_mask_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * math.pi * 0.5)


def mask_tokens(
    pred_tokens: torch.Tensor,
    mask_id: int,
    mask_prob: torch.Tensor,
    args: Namespace
) -> Tuple[torch.Tensor, torch.Tensor]:
    (batch_size, seq_len), device = pred_tokens.shape, pred_tokens.device
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
    mask_contiguous_region = False if args.mask_contiguous_region_prob is None else (random.random() < args.mask_contiguous_region_prob)
    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=device)
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())
            num_token_masked_height = random.randint(math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_))
            num_token_masked_height = min(num_token_masked_height, resolution)
            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)
            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)
            mask[batch_idx, start_idx_height: start_idx_height + num_token_masked_height, start_idx_width: start_idx_width + num_token_masked_width] = 1
        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)
    input_ids = torch.where(mask, mask_id, pred_tokens)
    labels = torch.where(mask, pred_tokens, ignore_id)
    return input_ids, labels


def mask_or_random_replace_tokens(
    image_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    mask_id: int,
    args: Namespace,
    mask_schedule: Callable[[torch.Tensor], torch.Tensor],
    is_training: bool,
    seed: int = None
) -> torch.Tensor:
    assert image_tokens.shape[0] == action_tokens.shape[0], "image_batch_size and action_batch_size must be the same"
    assert image_tokens.device == action_tokens.device, "image_tokens and action_tokens must be on the same device"
    batch_size, device = image_tokens.shape[0], image_tokens.device
    if not is_training and seed is not None:
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)
    if not is_training and args.eval_mask_ratios is not None:
        mask_prob = random.choices(args.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=device)
    else:
        timestemps = torch.rand(batch_size, device=device)
        mask_prob = mask_schedule(timestemps)
        mask_prob = mask_prob.clip(args.min_masking_rate)
    image_tokens, image_labels = mask_tokens(
        pred_tokens=image_tokens,
        mask_id=mask_id,
        mask_prob=mask_prob,
        args=args
    )
    action_tokens, action_labels = mask_tokens(
        pred_tokens=action_tokens,
        mask_id=mask_id,
        mask_prob=mask_prob,
        args=args
    )
    if not is_training and seed is not None:
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        random.setstate(python_rng_state)
    return (image_tokens, image_labels), (action_tokens, action_labels), mask_prob