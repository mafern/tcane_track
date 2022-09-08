
import matplotlib.pyplot as plt


def set_plot_rc():
    ### for white background...
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Avant Garde"]})
    plt.rc("savefig", facecolor="white")
    plt.rc("axes", facecolor="white")
    plt.rc("axes", labelcolor="dimgrey")
    plt.rc("axes", labelcolor="dimgrey")
    plt.rc("xtick", color="dimgrey")
    plt.rc("ytick", color="dimgrey")


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def format_spines(ax):
    adjust_spines(ax, ["left", "bottom"])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("dimgrey")
    ax.spines["bottom"].set_color("dimgrey")
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params("both", length=4, width=2, which="major", color="dimgrey")

#     ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)