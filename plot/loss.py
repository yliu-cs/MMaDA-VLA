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
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "PreTrained"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
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
    
    loss_path = os.path.join(args.ckpt_dir, "total_losses.npy")
    loss = np.load(loss_path).tolist()
    ax.plot(range(len(loss)), loss, color=colors[0][0], linestyle="-", alpha=0.6, zorder=1)
    smoothed_loss = gaussian_filter1d(loss, sigma=100)
    ax.plot(
        range(len(smoothed_loss))
        , smoothed_loss
        , color=colors[0][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Loss"
    )
    
    ax.set_xlim(left=0, right=len(loss))
    
    ax.set_ylim(
        bottom=list(sorted(loss, reverse=True))[len(loss) - 1] - 0.1,
        top=list(sorted(loss))[math.ceil(len(loss) * 0.9985)]
    )
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(30000))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10000)))
    
    ax.set_xlabel("Steps")
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
        plt.savefig(os.path.join(args.figure_dir, "loss.pdf"))
        if os.path.exists(os.path.join(args.figure_dir, "loss.png")):
            os.remove(os.path.join(args.figure_dir, "loss.png"))
    else:
        plt.savefig(os.path.join(args.figure_dir, "loss.png"), dpi=600)
    plt.close()


if __name__ == "__main__":
    args = get_args()
    main(args)