import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from argparse import Namespace, ArgumentParser


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.getcwd(), "ckpt"))
    parser.add_argument("--version", type=str, default=os.path.join("5afa0d69f888d1335985da8dbab36404"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "libero"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def main(args: Namespace):
    os.makedirs(args.figure_dir, exist_ok=True)

    colors = [
        ("#E1F2FC", "#5AA0F7"),
        ("#FBB3E5", "#E95A85"),
        ("#B1C5FD", "#7B87FF"),
        ("#C4B4E5", "#A373C8"),
    ]

    plt.rc("font", **{"family": "Times New Roman", "size": 12})
    fig, ax = plt.subplots(figsize=(7, 5))
    
    vision_loss_path = os.path.join(args.ckpt_dir, "MMaDA-VLA", args.version, "vision_losses.npy")
    vision_loss = np.load(vision_loss_path).tolist()
    ax.plot(range(len(vision_loss)), vision_loss, color=colors[0][0], linestyle="-", alpha=0.6, zorder=1)
    vision_smoothed_loss = gaussian_filter1d(vision_loss, sigma=100)
    ax.plot(
        range(len(vision_smoothed_loss))
        , vision_smoothed_loss
        , color=colors[0][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Goal Image Loss"
    )
    
    action_loss_path = os.path.join(args.ckpt_dir, "MMaDA-VLA", args.version, "action_losses.npy")
    action_loss = np.load(action_loss_path).tolist()
    ax.plot(range(len(action_loss)), action_loss, color=colors[1][0], linestyle="-", alpha=0.2, zorder=1)
    action_smoothed_loss = gaussian_filter1d(action_loss, sigma=100)
    ax.plot(
        range(len(action_smoothed_loss))
        , action_smoothed_loss
        , color=colors[1][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Action Loss"
    )
    
    max_len = max(len(vision_loss), len(action_loss))
    ax.set_xlim(left=0, right=max_len)
    
    cross_entropy_losses = vision_loss + action_loss
    ax.set_ylim(
        bottom=0.5,
        top=list(sorted(cross_entropy_losses))[math.ceil(len(cross_entropy_losses) * 0.992)]
    )
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(10000))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10000)))
    
    ax.set_xlabel("Steps (1e4)")
    ax.set_ylabel("Cross-Entropy Loss", color='black')
    
    for spine in ["top"]:
        ax.spines[spine].set_color("none")
    
    ax.grid(
        axis="y"
        , linestyle=(0, (5, 10))
        , linewidth=0.25
        , color="#4E616C"
        , zorder=-100
    )
    
    ax.legend(loc="upper right")
    
    if args.pdf:
        plt.savefig(os.path.join(args.figure_dir, "Loss.pdf"))
        if os.path.exists(os.path.join(args.figure_dir, "Loss.png")):
            os.remove(os.path.join(args.figure_dir, "Loss.png"))
    else:
        plt.savefig(os.path.join(args.figure_dir, "Loss.png"), dpi=600)
    plt.close()


if __name__ == "__main__":
    args = get_args()
    main(args)