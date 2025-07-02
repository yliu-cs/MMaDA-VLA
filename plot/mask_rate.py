import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from argparse import Namespace, ArgumentParser


# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.getcwd(), "ckpt"))
    parser.add_argument("--version", type=str, default=os.path.join("ca1d09542fde601afad882bfb4e2fdff"))
    parser.add_argument("--figure_dir", type=str, default=os.path.join(os.getcwd(), "figure"))
    parser.add_argument("--pdf", action="store_true", help="Export PDF")
    args = parser.parse_args()
    return args


def plot_mask_rate_density(mask_rates, args):
    """Plot mask rate density distribution"""
    plt.rc("font", **{"family": "Times New Roman", "size": 12})
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot histogram with density
    ax.hist(mask_rates, bins=50, density=True, alpha=0.7, color="#FBB3E5", edgecolor='black', zorder=1)
    ax.set_xlim(left=0, right=1.0)
    # Set ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_xlabel("Mask Rate", color='black')
    ax.set_ylabel("Density", color='black')
    ax.set_title("Mask Rate Density Distribution")
    # Remove top and right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_color("none")
    # Add grid
    ax.grid(axis="y", linestyle=(0, (5, 10)), linewidth=0.25, color="#4E616C", zorder=-100)
    # Save figure
    if args.pdf:
        plt.savefig(os.path.join(args.figure_dir, "MaskRate_Density.pdf"))
        if os.path.exists(os.path.join(args.figure_dir, "MaskRate_Density.png")):
            os.remove(os.path.join(args.figure_dir, "MaskRate_Density.png"))
    else:
        plt.savefig(os.path.join(args.figure_dir, "MaskRate_Density.png"), dpi=600)
    plt.close()


def plot_mask_rate_boxplot(mask_rates, args):
    """Plot mask rate box plot"""
    plt.rc("font", **{"family": "Times New Roman", "size": 12})
    fig, ax = plt.subplots(figsize=(7, 5))
    # Create box plot
    box_plot = ax.boxplot(
        mask_rates,
        patch_artist=True,
        boxprops={"facecolor": "#B1C5FD", "alpha": 0.7},
        medianprops={"color": "#7B87FF", "linewidth": 2},
        flierprops={"marker": 'o', "markerfacecolor": "#7B87FF", "markersize": 3}
    )
    ax.set_title("Mask Rate Distribution")
    ax.set_ylabel("Mask Rate", color='black')
    # Remove top and right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_color("none")
    # Add grid
    ax.grid(axis="y", linestyle=(0, (5, 10)), linewidth=0.25, color="#4E616C", zorder=-100)
    # Add statistical information
    q25, q75 = np.percentile(mask_rates, [25, 75])
    iqr = q75 - q25
    mean_val = np.mean(mask_rates)
    median_val = np.median(mask_rates)
    stats_text = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nQ1: {q25:.3f}\nQ3: {q75:.3f}\nIQR: {iqr:.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    # Save figure
    if args.pdf:
        plt.savefig(os.path.join(args.figure_dir, "MaskRate_BoxPlot.pdf"))
        if os.path.exists(os.path.join(args.figure_dir, "MaskRate_BoxPlot.png")):
            os.remove(os.path.join(args.figure_dir, "MaskRate_BoxPlot.png"))
    else:
        plt.savefig(os.path.join(args.figure_dir, "MaskRate_BoxPlot.png"), dpi=600)
    plt.close()


def main(args: Namespace):
    """Main function to plot mask rate statistics"""
    os.makedirs(args.figure_dir, exist_ok=True)
    # Load data
    mask_rates_path = os.path.join(args.ckpt_dir, "RoLD", args.version, "mask_rates.npy")
    if not os.path.exists(mask_rates_path):
        print(f"Error: File {mask_rates_path} does not exist")
        return
    # Load mask rate data
    mask_rates = np.load(mask_rates_path)
    print(f"Data shape: {mask_rates.shape}")
    print(f"Data type: {mask_rates.dtype}")
    print(f"Minimum: {mask_rates.min():.4f}")
    print(f"Maximum: {mask_rates.max():.4f}")
    print(f"Mean: {mask_rates.mean():.4f}")
    print(f"Standard deviation: {mask_rates.std():.4f}")
    print(f"Median: {np.median(mask_rates):.4f}")
    # Plot density distribution
    plot_mask_rate_density(mask_rates, args)
    print(f"Density plot saved to: {os.path.join(args.figure_dir, 'MaskRate_Density.png')}")
    # Plot box plot
    plot_mask_rate_boxplot(mask_rates, args)
    print(f"Box plot saved to: {os.path.join(args.figure_dir, 'MaskRate_BoxPlot.png')}")
    # Print detailed statistics
    print("\n=== Detailed Statistics ===")
    print(f"Number of data points: {len(mask_rates)}")
    print(f"25th percentile: {np.percentile(mask_rates, 25):.4f}")
    print(f"75th percentile: {np.percentile(mask_rates, 75):.4f}")
    print(f"90th percentile: {np.percentile(mask_rates, 90):.4f}")
    print(f"95th percentile: {np.percentile(mask_rates, 95):.4f}")
    print(f"99th percentile: {np.percentile(mask_rates, 99):.4f}")
    # Statistics for different mask rate ranges
    ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
              (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print("\n=== Mask Rate Range Distribution ===")
    for low, high in ranges:
        count = np.sum((mask_rates >= low) & (mask_rates < high))
        percentage = count / len(mask_rates) * 100
        print(f"[{low:.1f}, {high:.1f}): {count} points ({percentage:.1f}%)")


if __name__ == "__main__":
    args = get_args()
    main(args)