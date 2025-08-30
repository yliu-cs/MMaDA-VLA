import os
import re
import math
import torch
import autoroot
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from rold.models.actrvq import ActionRVQModel
from argparse import ArgumentParser, Namespace
from rold.train.train_actrvq import TrainingArguments


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "CALVIN"))
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--pretrained_actrvq", type=str, default=os.path.join(os.getcwd(), "ckpt", "ActRVQ_8steps"))
    parser.add_argument("--tqdm_flag", action="store_false")
    return parser.parse_args()


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrained_actrvqs = os.listdir(args.pretrained_actrvq)
    num_action_chunk = int(re.search(r'(\d+)steps', os.path.basename(args.pretrained_actrvq)).group(1))
    actions = np.load(os.path.join(args.data_path, "validation", f"calvin_abcd_d_{num_action_chunk}steps_action.npy"), allow_pickle=True)
    
    results = []
    for pretrained_actrvq in tqdm(pretrained_actrvqs):
        try:
            act_rvq_model = ActionRVQModel.from_pretrained(os.path.join(args.pretrained_actrvq, pretrained_actrvq)).to(device).eval()
        except:
            continue
        pretrained_args = torch.load(os.path.join(args.pretrained_actrvq, pretrained_actrvq, "training_args.bin"), weights_only=False)
        used_ids, mses = set(), []
        for i in tqdm(range(math.ceil(len(actions) / args.batch_size)), disable=args.tqdm_flag):
            action = actions[i * args.batch_size:min((i + 1) * args.batch_size, len(actions))]
            action = torch.tensor(action, dtype=torch.float, device=device)
            act_ids = act_rvq_model.tokenize(action)
            recon_action = act_rvq_model.detokenize(act_ids)
            act_ids = act_ids.flatten().cpu().numpy().tolist()
            used_ids.update(act_ids)
            mses.append(F.mse_loss(action, recon_action).item())
        results.append({
            "id": pretrained_actrvq,
            "lr": str(pretrained_args.learning_rate),
            "dims": str(act_rvq_model.config.dims),
            "levels": str(act_rvq_model.config.levels),
            "num_quan": str(act_rvq_model.config.num_quantizers),
            "vocab_size": str(act_rvq_model.config.codebook_size),
            "ratio": f"{str(len(used_ids)):>5}/{act_rvq_model.config.codebook_size:<5}",
            "rate": round(len(used_ids)/act_rvq_model.config.codebook_size*100, 2),
            "mse": np.mean(mses),
        })
    for result in sorted(results, key=lambda x: x["mse"]):
        print_str = " | ".join([
            f"{result['id']}",
            f"LR: {result['lr']:<8}",
            f"Dims: {result['dims']:<28}",
            f"Levels: {result['levels']:<15}",
            f"Num. Quantizers: {result['num_quan']:<2}",
            f"Codebook Size: {result['vocab_size']:<5}",
            f"Utility Ratio: {result['ratio']:<5}({str(result['rate']):>5}%)",
            f"MSE: {result['mse']:.4f}"
        ])
        print(f"{'❌' if (result['rate'] < 85 or result['mse'] > 0.01) else '✅'} {print_str}")


if __name__ == "__main__":
    args = get_args()
    main(args)