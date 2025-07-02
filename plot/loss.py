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
    parser.add_argument("--version", type=str, default=os.path.join("ca1d09542fde601afad882bfb4e2fdff"))
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
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    
    vision_loss_path = os.path.join(args.ckpt_dir, "RoLD", args.version, "vision_losses.npy")
    vision_loss = np.load(vision_loss_path).tolist()
    ax1.plot(range(len(vision_loss)), vision_loss, color=colors[0][0], linestyle="-", alpha=0.2, zorder=1)
    vision_smoothed_loss = gaussian_filter1d(vision_loss, sigma=100)
    ax1.plot(
        range(len(vision_smoothed_loss))
        , vision_smoothed_loss
        , color=colors[0][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Goal Image Discrete Loss"
    )
    
    action_loss_path = os.path.join(args.ckpt_dir, "RoLD", args.version, "action_losses.npy")
    action_loss = np.load(action_loss_path).tolist()
    ax1.plot(range(len(action_loss)), action_loss, color=colors[1][0], linestyle="-", alpha=0.2, zorder=1)
    action_smoothed_loss = gaussian_filter1d(action_loss, sigma=100)
    ax1.plot(
        range(len(action_smoothed_loss))
        , action_smoothed_loss
        , color=colors[1][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Action Discrete Loss"
    )
    
    mse_loss_path = os.path.join(args.ckpt_dir, "RoLD", args.version, "mses.npy")
    mse_loss = np.load(mse_loss_path).tolist()
    ax2.plot(range(len(mse_loss)), mse_loss, color=colors[2][0], linestyle="-", alpha=0.2, zorder=1)
    mse_smoothed_loss = gaussian_filter1d(mse_loss, sigma=100)
    ax2.plot(
        range(len(mse_smoothed_loss))
        , mse_smoothed_loss
        , color=colors[2][1]
        , linestyle="-"
        , linewidth=2
        , zorder=100
        , label="Action Continuous MSE"
    )
    
    max_len = max(len(vision_loss), len(action_loss), len(mse_loss))
    ax1.set_xlim(left=0, right=max_len)
    
    cross_entropy_losses = vision_loss + action_loss
    ax1.set_ylim(
        bottom=0.5,
        top=list(sorted(cross_entropy_losses))[math.ceil(len(cross_entropy_losses) * 0.992)]
    )
    
    ax2.set_ylim(
        bottom=0,
        top=list(sorted(cross_entropy_losses))[math.ceil(len(cross_entropy_losses) * 0.992)] / 20
    )
    
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10000))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x // 10000)))
    
    ax1_ticks = ax1.get_yticks()
    ax1_min, ax1_max = ax1.get_ylim()
    
    valid_ax1_ticks = [tick for tick in ax1_ticks if ax1_min <= tick <= ax1_max]
    
    ax2_values = [0.01 * i for i in range(len(valid_ax1_ticks))]
    
    if len(valid_ax1_ticks) >= len(ax2_values):
        target_positions = valid_ax1_ticks[:len(ax2_values)]
        
        ax2_min_pos = target_positions[0]  # 第一个刻度位置
        ax2_max_pos = target_positions[-1]  # 最后一个刻度位置
        
        ax2_min_val = ax2_values[0]  # 0.0
        ax2_max_val = ax2_values[-1]  # 0.08
        
        position_range = ax2_max_pos - ax2_min_pos
        value_range = ax2_max_val - ax2_min_val
        
        ax2_bottom = ax2_min_val - (ax2_min_pos - ax1_min) * value_range / position_range
        ax2_top = ax2_max_val + (ax1_max - ax2_max_pos) * value_range / position_range
        
        ax2.set_ylim(ax2_bottom, ax2_top)
        ax2.set_yticks(ax2_values)
    
    ax1.set_xlabel("Steps (1e4)")
    ax1.set_ylabel("Cross-Entropy Loss", color='black')
    ax2.set_ylabel("MSE", color='black')
    
    ax2.tick_params(axis='y', labelcolor='black')
    
    for spine in ["top"]:
        ax1.spines[spine].set_color("none")
        ax2.spines[spine].set_color("none")
    
    ax1.grid(
        axis="y"
        , linestyle=(0, (5, 10))
        , linewidth=0.25
        , color="#4E616C"
        , zorder=-100
    )
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
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