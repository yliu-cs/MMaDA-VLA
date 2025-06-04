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
    parser.add_argument("--version", type=str, default="620e690f0e7f7b7ba833357f72eb1807")
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def main(args: Namespace):
    os.makedirs(args.figure_dir, exist_ok=True)

    colors = [
        ("#E1F2FC", "#5AA0F7")
        , ("#B1C5FD", "#7B87FF")
        , ("#FBB3E5", "#E95A85")
        , ("#C4B4E5", "#A373C8")
    ]

    plt.rc("font", **{"family": "Times New Roman", "size": 12})
    fig, ax = plt.subplots()
    loss_path = os.path.join(args.ckpt_dir, "RoLD", args.version, "losses.npy")
    loss = np.load(loss_path).tolist()
    i = random.sample(range(len(colors)), 1)[0]
    ax.plot(range(len(loss)), loss, color=colors[i][0], linestyle="-", alpha=0.5, zorder=1)
    smoothed_loss = gaussian_filter1d(loss, sigma=50)
    ax.plot(
        range(len(smoothed_loss))
        , smoothed_loss
        , color=colors[i][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
    )
    ax.set_xlim(left=0, right=len(loss))
    ax.set_ylim(bottom=min(loss) - 0.3, top=sorted(loss)[math.ceil(len(loss) * 0.992)])
    ax.xaxis.set_major_locator(plt.MultipleLocator(2000))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 2000)))
    ax.set_xlabel("Steps (2e3)")
    ax.set_ylabel("Loss")
    for spine in ["top", "right"]:
        ax.spines[spine].set_color("none")
    ax.grid(
        axis="y"
        , linestyle=(0, (5, 10))
        , linewidth=0.25
        , color="#4E616C"
        , zorder=-100
    )
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