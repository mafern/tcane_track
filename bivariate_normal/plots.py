import matplotlib.pyplot as plt
import cartopy as ct
import cartopy.feature as cfeature
import mahalanobis
import cmasher as cmr
import data_info

import numpy as np

DATA_CRS = ct.crs.PlateCarree()
KM_TO_DEG = 1. / 111


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

def draw_coastlines(ax):
    # ADD COASTLINES
    # ax.set_global()
    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor=(.95, .95, .95),
        edgecolor='k',
        linewidth=.5,
        zorder=0,
    )
    ax.add_feature(land_feature)


def plot_leadtime_predictions(df,
                              ax,
                              leadtimes=(24, 48, 72, 96, 120),
                              contours=np.arange(.1, 1., .1),
                              ):
    COLOR = cmr.take_cmap_colors("cmr.pride", len(contours), cmap_range=(0.2, 0.8), return_fmt="hex")

    for lead_time in leadtimes:
        df_plot = df.loc[(df["ftime(hr)"] == lead_time)]
        if df_plot.empty:
            print(str(lead_time) + ' dataframe is empty')
            continue

        besttrack_u = df_plot["LONC"].values + KM_TO_DEG * df_plot["OBDX"].values
        besttrack_v = df_plot["LATC"].values + KM_TO_DEG * df_plot["OBDY"].values

        mahalanobis.plot_cdf(
            df_plot["LONC"].values + KM_TO_DEG * df_plot["mu_u"].values,
            df_plot["LATC"].values + KM_TO_DEG * df_plot["mu_v"].values,
            KM_TO_DEG * df_plot["sigma_u"].values,
            KM_TO_DEG * df_plot["sigma_v"].values,
            df_plot["rho"].values,
            besttrack_u=besttrack_u,
            besttrack_v=besttrack_v,
            colors=COLOR,
            contours=contours,
            data_crs=DATA_CRS,
        )


        # plot consensus prediction
        plt.plot(df_plot["LONC"].values,
                 df_plot["LATC"].values,
                 marker='o',
                 markerfacecolor='None',
                 markeredgewidth=.25,
                 linestyle='',
                 markersize=3,
                 color='k',
                 label='Consensus',
                 transform=DATA_CRS,
                 )

        plt.text(df_plot["LONC"].values,
                 df_plot["LATC"].values,
                 df_plot['ftime(hr)'].values[0],
                 color="k",
                 fontsize=8,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=DATA_CRS,
                 )

    # connect the besttrack predictions
    plt.plot(
        df["LONC"].values + KM_TO_DEG * df["OBDX"].values,
        df["LATC"].values + KM_TO_DEG * df["OBDY"].values,
        "-",
        linewidth=.25,
        color="k",
        transform=DATA_CRS,
    )

    # format plot
    format_spines(ax)

    # set grid lines
    gl = ax.gridlines(crs=ct.crs.PlateCarree(), draw_labels=True,
                      linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False

    # setup legend
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[:1], labels[:1], loc=2)

    # set storm title
    details = data_info.get_storm_details(df, 0)
    details = details[:details.rfind(' @')]
    plt.title(details)

    draw_coastlines(ax)

    return details
