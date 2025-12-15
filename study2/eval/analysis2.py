import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

import pandas as pd
from study2.eval.process import DataPoint,TorchPolicy
from study2.eval.eval_utils import get_processed_fnames
from study2.eval.gather_csv import gather
from study2.static import *


##############################################
############### Style Dicts ##################
##############################################
charcoal = (57/255, 57/255, 57/255)
style_dict = {}
color_dict = {'rs-tom': (255/255, 103/255, 1/255),
              'rational': charcoal,
              0: charcoal,
              1: charcoal
              }

line_style = {'rs-tom': {'ls':'-', 'lw':2, 'color':color_dict['rs-tom'], 'marker':'.','markersize':10},
              'rational': {'ls':':', 'lw':2, 'color':color_dict['rational'], 'marker':'.','markersize':10}, #,'markerfacecolor':'white'
              0: {'ls':'-'},
                1: {'ls':':','markerfacecolor':'white'}
              }
bar_style = {'rs-tom': {'color':color_dict['rs-tom'], 'edgecolor':'w'},
              'rational': {'color':color_dict['rational'], 'edgecolor':'w'},
              0: {'color':color_dict[0], 'edgecolor':'w'},
              1: {'color':color_dict[1],'hatch':'//', 'edgecolor':'w'},
              'other': {'hatch': ['','//','\\\\','++','xx','..']
                        }
              }


global_params = {'legend': {'fontsize': 8, 'facecolor':'lightgray', 'framealpha':0.8, 'frameon':True}}
##############################################
############### Data Helpers #################
##############################################
def load_data():
    fname_dict = get_processed_fnames(full_path=False)
    COND1_FNAMES = fname_dict['cond1']
    COND0_FNAMES = fname_dict['cond0']

    dps_cond1 = []
    dps_cond0 = []
    dps_all = []
    for fname in COND0_FNAMES:
        fpath = PROCESSED_COND0_DIR + "\\" + fname
        dp = DataPoint.load_processed(fpath)
        dps_cond0.append(dp)

    for fname in COND1_FNAMES:
        fpath = PROCESSED_COND1_DIR + "\\" + fname
        dp = DataPoint.load_processed(fpath)
        dps_cond1.append(dp)

    dps_all += dps_cond0 + dps_cond1
    return dps_cond0, dps_cond1, dps_all

def split_by_partner(data_points, metric: str):
    rs_tom, rational = [], []
    for dp in data_points:
        rs_tom.append(eval(f"dp.{metric}['rs-tom']"))
        rational.append(eval(f"dp.{metric}['rational']"))
    return rs_tom, rational

##############################################
############### Plotters #####################
##############################################
def plot_timeseries(ax, data, measure, scatter=False, boxplot=False, title='', ylabel=''):
    N_games = 10
    N_pgames = int(N_games / 2)
    capsize = 5
    offsets = {'rs-tom': -0.15,'rational': 0.15}
    conditions = [['rs-tom', 'rational'], ['rational', 'rs-tom']]
    cond_plot_order = [0,1]  # condition 0 first, then condition 1
    n = -1
    legend_params = dict(frameon=False, loc='upper right', ncol=2 )
    legend_params.update(**global_params['legend'])

    # Plot separate games and partners
    for icond in cond_plot_order:
        n += 1
        partners = conditions[icond]
        for ipartner in range(2):
            partner = partners[ipartner]
            df = data[data['condition'] == icond]
            df = df[df['partner_type'] == partner]

            game_means, game_stds = [], []
            this_style = copy.deepcopy(line_style[partner])
            this_style.update(**line_style[icond])


            game_nums = np.arange(N_pgames*(ipartner) , N_pgames*((ipartner+1)) )
            for i in game_nums:
                game_data = df[df['game_num'] == i]
                game_mean = game_data[measure].mean()
                game_std = game_data[measure].std()
                game_means.append(game_mean)
                game_stds.append(game_std)

            if not boxplot:
                x = game_nums + offsets[partner]
                means = np.array(game_means)
                stds = np.array(game_stds)
                conf95 = 1.98 * stds# / means
                if measure == 'C-ACT':
                    pass



                # if n==0:
                #     ax.plot(x, means, **this_style, label=partner)
                # else:
                #     ax.plot(x, means, **this_style)
                ax.errorbar(x, means, yerr=conf95,capsize=capsize,alpha=0.75, **this_style)
                label = f"C{icond}, {partner}"
                ax.plot(x, means, **this_style, label=label)

    # Overlay scatter points ##############
    if scatter:

        for icond in range(2):
            partners = conditions[icond]
            for ipartner in range(2):
                partner = partners[ipartner]
                df = data[data['condition'] == icond]
                df = df[df['partner_type'] == partner]
                game_nums = np.arange(int(N_games / 2) * (ipartner), int(N_games / 2) * ((ipartner + 1) * 2))

                for i in game_nums:
                    game_data = df[df['game_num'] == i]
                    x_vals = np.full(len(game_data), i + offsets[partner])
                    # x_vals = + np.random.uniform(-0.05, 0.05, size=len(game_data))
                    y_vals = game_data[measure].values
                    ax.scatter(x_vals, y_vals, color=line_style[partner]['color'], alpha=0.3, s=10)

    # ------------------------------------------------------------------
    # Annotate bottom of ax with partner types (1 or 2) with filled box
    # ------------------------------------------------------------------
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax)  # add 10% headroom for partner boxes
    # Get current y-limits after plotting
    ymin, ymax = ax.get_ylim()
    box_height = 0.06 * (ymax - ymin)  # 6% of the y-range

    half = N_games // 2  # first 5 games, second 5 games

    # Partner 1 box: games 0–4 (xticks G1–G5)
    rect1 = Rectangle(
        (-0.5, ymin),  # start slightly before first tick
        half,  # width in data units (5 games)
        box_height,  # height
        facecolor='lightgray',
        edgecolor='none',
        alpha=0.4,
        zorder=0  # behind data
    )
    ax.add_patch(rect1)
    ax.text(
        -0.5 + half / 2,
        ymin + box_height / 2,
        "Partner 1",
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

    # Partner 2 box: games 5–9 (xticks G1–G5 again)
    rect2 = Rectangle(
        (-0.5 + half, ymin),  # starts at x=4.5
        half,
        box_height,
        facecolor='darkgray',
        edgecolor='none',
        alpha=0.4,
        zorder=0
    )
    ax.add_patch(rect2)
    ax.text(
        -0.5 + half + half / 2,
        ymin + box_height / 2,
        "Partner 2",
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )
    # ------------------------------------------------------------------

    # Customize Axes
    xticks = [f'G{i+1}' for i in range(int(N_games/2))]*2
    ax.set_xticks(np.arange(N_games))
    ax.set_xticklabels(xticks)

    ax.set_xlabel("Game Number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(conditions[0], loc='upper right')
    ax.legend(**legend_params)


def plot_grandmean_box(ax, data, measures: list, by_condition=True, by_partner=True, independent_scale=None, ylabel=None):
    """Plots grouped boxplots for the specified measures.

    [by_condition, by_partner] = [True, False]: boxplots for each condition (2 groups per measure)
                                - cond0
                                - cond1
    [by_condition, by_partner] = [False, True]: boxplots for each partner (2 groups per measure)
                                - rs-tom
                                - rational
    [by_condition, by_partner] = [True, True]:  boxplots for each condition x partner cell (up to 4 groups per measure)
                                - cond0, rs-tom
                                - cond0, rational
                                - cond1, rs-tom
                                - cond1, rational

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the plot on.
    data : pandas.DataFrame
        Must contain 'condition', 'partner_type', and the given measures.
    measures : list of str
        Column names of dependent variables to plot.
    by_condition : bool, default True
        Whether to group by condition.
    by_partner : bool, default True
        Whether to group by partner_type.
    independent_scale : list of str or None
        Subset of `measures` that should be plotted on a secondary y-axis
        with an independent scale.
    """

    if not (by_condition or by_partner):
        raise ValueError("At least one of by_condition or by_partner must be True.")

    ylabel = ylabel or "Value"

    # Normalize independent_scale handling early
    independent_names = list(independent_scale or [])
    independent_set = set(independent_names)

    condition_col = "condition"
    partner_col = "partner_type"

    # Decide which columns define groups
    group_cols = []
    if by_condition:
        group_cols.append(condition_col)
    if by_partner:
        group_cols.append(partner_col)

    # Group by condition/partner combination (no aggregation yet; we want distributions)
    grouped = data.groupby(group_cols)
    group_keys = list(grouped.groups.keys())
    n_groups = len(group_keys)
    n_measures = len(measures)

    # Horizontal layout: clusters of groups per measure
    width = 0.8 / max(n_groups, 1)  # width of a single group's box within a cluster
    x = np.arange(n_measures, dtype=float)

    # If some measures use an independent scale, offset those measures
    if len(independent_names) >= 1:
        independent_offset = (width * n_groups) / 2.0
        x[-len(independent_names):] += independent_offset

    # Secondary axis for independent-scale measures, if any of them are present in measures
    use_secondary = any(m in independent_set for m in measures)
    ax2 = ax.twinx() if use_secondary else None

    legend_patches = {}

    for g_idx, key in enumerate(group_keys):
        # key can be scalar (single group col) or tuple (multi-col)
        orig_key = key
        key_tuple = key if isinstance(key, tuple) else (key,)

        # Human-readable label for legend
        label_parts = []
        for col, val in zip(group_cols, key_tuple):
            if col == condition_col and val in [0, 1]:
                label_parts.append(f"C{val}")
            else:
                label_parts.append(str(val))
        label = ", ".join(label_parts)

        # Style lookup using existing bar_style mapping
        if by_partner and not by_condition:
            base_style = dict(bar_style[key_tuple[0]])  # partner_type
        elif by_condition and not by_partner:
            base_style = dict(bar_style[key_tuple[0]])  # condition
        elif by_condition and by_partner:
            # Combine styles (condition first, then partner)
            base_style = dict(bar_style[key_tuple[0]])
            base_style.update(bar_style[key_tuple[1]])
        else:
            raise ValueError("At least one of by_condition or by_partner must be True.")

        # Derive a usable color/alpha for the boxplot from the style dict
        color = (base_style.get("color") or
                 base_style.get("edgecolor") or
                 base_style.get("facecolor") or
                 f"C{g_idx}")
        alpha = base_style.get("alpha", 0.6)

        # Horizontal offset for this group within each measure cluster
        group_offset = (g_idx - (n_groups - 1) / 2.0) * width

        # Add a legend proxy once per group
        if label not in legend_patches:
            legend_patches[label] = Patch(facecolor=color,
                                          edgecolor="black",
                                          alpha=alpha,
                                          label=label)

        # Data for this group
        group_df = grouped.get_group(orig_key)

        # One box per (group, measure)
        for m_idx, m_name in enumerate(measures):
            vals = group_df[m_name].dropna().values
            if vals.size == 0:
                continue  # nothing to plot

            x_pos = x[m_idx] + group_offset
            target_ax = ax2 if (ax2 is not None and m_name in independent_set) else ax

            bp = target_ax.boxplot(
                [vals],
                positions=[x_pos],
                widths=width * 0.9,
                patch_artist=True,
                manage_ticks=False,
            )

            # Style boxplot elements
            for box in bp["boxes"]:
                box.set(facecolor=color, edgecolor=color, alpha=alpha)
            for whisker in bp["whiskers"]:
                whisker.set(color=color)
            for cap in bp["caps"]:
                cap.set(color=color)
            for median in bp["medians"]:
                median.set(color="black", linewidth=1.5)
            for flier in bp["fliers"]:
                flier.set(marker="o", markersize=3, markerfacecolor=color,
                          markeredgecolor=color, alpha=alpha)

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.margins(x=0.05)

    # Y-axis labels
    ax.set_ylabel(ylabel)
    if ax2 is not None:
        if len(independent_names) == 1:
            y2_label = independent_names[0]
        else:
            y2_label = "Value (" + ", ".join(independent_names) + ")"
        ax2.set_ylabel(y2_label)

    # Legend
    if legend_patches:
        handles = list(legend_patches.values())
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, frameon=False, loc="upper left")

    # Attach secondary axis for external use without breaking return type
    ax.independent_ax = ax2
    return ax
def plot_grandmean_bar(ax, data, measures: list,
                       by_condition=True, by_partner=True,
                       independent_scale=None, ylim=None,
                       ylabel=None, title='',zero_line=False):
    """Plots grouped bar charts for the specified measures.
    [by_condition, by_partner] = [True,False]: plot grouped bars for each condition mean (2 groups per measure)
                                - mean for cond0
                                - mean for cond1
    [by_condition, by_partner] = [False,True]: plot grouped bars for each partner mean (2 groups per measure)
                                - mean for rs-tom
                                - mean for rational
    [by_condition, by_partner] = [True,True]:  plot grouped bars for each condition x partner mean (4 groups per measure)
                                - mean for cond0, rs-tom
                                - mean for cond0, rational
                                - mean for cond1, rs-tom
                                - mean for cond1, rational

    """

    if not (by_condition or by_partner):
        raise ValueError("At least one of by_condition or by_partner must be True.")
    ylabel = ylabel or "Mean Value"
    legend_params = dict(frameon=False, loc='upper left', ncol=2 if (by_condition and by_partner) else 1)
    legend_params.update(**global_params['legend'])
    ax.set_title(title)
    condition_col = "condition"
    partner_col = "partner_type"

    # Decide which columns define groups
    group_cols = []
    if by_condition:
        group_cols.append(condition_col)
    if by_partner:
        group_cols.append(partner_col)


    # Aggregate means for all requested measures
    grouped = data.groupby(group_cols)[measures].mean()



    # Group index (single index or MultiIndex)
    groups = list(grouped.index)
    n_groups = len(groups)
    n_measures = len(measures)

    if by_partner and by_condition:
        groups[0] = (0, 'rs-tom')
        groups[1] = (0, 'rational')



    width = 0.8 / max(n_groups, 1)  # total bar cluster width ~0.8


    x = np.arange(n_measures).astype(float)
    if independent_scale is not None and len(independent_scale) >= 1:
        independent_offset = (width * n_groups)/2
        x[-len(independent_scale):] += independent_offset

    # Handle independent-scale measures
    independent_scale = set(independent_scale or [])
    use_secondary = any(m in independent_scale for m in measures)
    ax2 = ax.twinx() if use_secondary else None

    if zero_line:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Plot one bar series per group combination
    for g_idx, key in enumerate(groups):
        vals = grouped.loc[key].values
        if np.ndim(vals) == 0:  # single measure edge case
            vals = np.array([vals])

        # Normalize key to a tuple for consistent labeling
        if not isinstance(key, tuple):
            key = (key,)

        label_parts = [f"C{val}" if val in [0, 1] else f"{val}" for col, val in zip(group_cols, key)]
        label = ", ".join(label_parts)

        group_offset = (g_idx - (n_groups - 1) / 2) * width


        if by_partner and not by_condition:
            this_style = bar_style[key[0]]
        elif by_condition and not by_partner:
            this_style = bar_style[key[0]]
        elif by_condition and by_partner:
            this_style = bar_style[key[0]]
            this_style.update(**bar_style[key[1]])
        else:
            raise ValueError("At least one of by_condition or by_partner must be True.")


        for m_idx, (m_name, val) in enumerate(zip(measures, vals)):
            x_pos = x[m_idx] + group_offset
            target_ax = ax2 if (ax2 is not None and m_name in independent_scale) else ax

            # Only use the label once per group to avoid legend clutter
            target_ax.bar( x_pos, val, width,
                label=label if m_idx == 0 else "_nolegend_",
                **this_style
            )

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.margins(x=0.05)
    ylim = ylim or ax.get_ylim()
    ax.set_ylim(*ylim)

    # Y-axis labels
    ax.set_ylabel(ylabel)
    if ax2 is not None:
        if len(independent_scale) == 1:
            y2_label = list(independent_scale)[0]
        else:
            y2_label = "(" + ", ".join(independent_scale) + ")"
        ax2.set_ylabel(y2_label)

    # Combine legends from both axes if needed
    handles1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Remove "_nolegend_" entries
        combined = [(h, l) for h, l in (list(zip(handles1, labels1)) +
                                        list(zip(handles2, labels2)))
                    if l != "_nolegend_"]
        if combined:
            handles, labels = zip(*combined)

            ax.legend(handles, labels, **legend_params)
    else:
        # Clean legend for single axis
        filtered = [(h, l) for h, l in zip(handles1, labels1) if l != "_nolegend_"]
        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(handles, labels, **legend_params)

    # Attach secondary axis for external use without breaking return type
    ax.independent_ax = ax2
    return ax


def plot_grandmean_bar2(ax, data, measures: list, groupby,
                       independent_scale=None,
                       ylabel=None, title=''):
    """Plots grouped bar charts for the specified measures.
    [by_condition, by_partner] = [True,False]: plot grouped bars for each condition mean (2 groups per measure)
                                - mean for cond0
                                - mean for cond1
    [by_condition, by_partner] = [False,True]: plot grouped bars for each partner mean (2 groups per measure)
                                - mean for rs-tom
                                - mean for rational
    [by_condition, by_partner] = [True,True]:  plot grouped bars for each condition x partner mean (4 groups per measure)
                                - mean for cond0, rs-tom
                                - mean for cond0, rational
                                - mean for cond1, rs-tom
                                - mean for cond1, rational

    """




    # -------------------------------

    assert isinstance(groupby, (str, list)), "groupby must be a string or list of strings."

    if isinstance(groupby, str): groupby = [groupby]
    by_partner = 'partner_type' in groupby
    by_condition = 'condition' in groupby
    by_other = len(groupby) - sum([by_partner, by_condition]) > 0

    # verify groupby columns exist in data
    for col in groupby:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data for grouping.")

    # -------------------------------

    ylabel = ylabel or "Mean Value"
    ax.set_title(title)

    # Aggregate means for all requested measures
    grouped = data.groupby(groupby)[measures].mean()

    # Group index (single index or MultiIndex)
    groups = list(grouped.index)
    n_groups = len(groups)
    n_measures = len(measures)

    width = 0.8 / max(n_groups, 1)  # total bar cluster width ~0.8

    legend_params = dict(frameon=False, loc='upper left',
                         ncol=2 if n_groups>=4 else 1
                         )
    legend_params.update(**global_params['legend'])

    x = np.arange(n_measures).astype(float)
    if independent_scale is not None and len(independent_scale) >= 1:
        independent_offset = (width * n_groups)/2
        x[-len(independent_scale):] += independent_offset

    # Handle independent-scale measures
    independent_scale = set(independent_scale or [])
    use_secondary = any(m in independent_scale for m in measures)
    ax2 = ax.twinx() if use_secondary else None

    # Plot one bar series per group combination
    for g_idx, key in enumerate(groups):
        vals = grouped.loc[key].values
        if np.ndim(vals) == 0:  # single measure edge case
            vals = np.array([vals])

        # Normalize key to a tuple for consistent labeling
        if not isinstance(key, tuple):
            key = (key,)

        label_parts = [f"C{val}" if val in [0, 1] else f"{val}" for col, val in zip(groupby, key)]
        label = ", ".join(label_parts)

        group_offset = (g_idx - (n_groups - 1) / 2) * width


        if by_partner and not by_condition:
            this_style = copy.deepcopy(bar_style[key[0]])
        elif by_condition and not by_partner:
            this_style = copy.deepcopy(bar_style[key[0]])
        elif by_condition and by_partner:
            this_style = copy.deepcopy(bar_style[key[0]])
            this_style.update(**bar_style[key[1]])
        else:
            raise ValueError("At least one of by_condition or by_partner must be True.")

        if by_other:
            _n_other = int(len(groups)/(2 * (int(by_partner) + int(by_condition))))
            i_other = g_idx % _n_other
            this_style.update(hatch = bar_style['other']['hatch'][i_other])

        for m_idx, (m_name, val) in enumerate(zip(measures, vals)):
            x_pos = x[m_idx] + group_offset
            target_ax = ax2 if (ax2 is not None and m_name in independent_scale) else ax

            # Only use the label once per group to avoid legend clutter
            target_ax.bar( x_pos, val, width,
                label=label if m_idx == 0 else "_nolegend_",
                **this_style
            )

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.margins(x=0.05)

    # Y-axis labels
    ax.set_ylabel(ylabel)
    if ax2 is not None:
        if len(independent_scale) == 1:
            y2_label = list(independent_scale)[0]
        else:
            y2_label = "(" + ", ".join(independent_scale) + ")"
        ax2.set_ylabel(y2_label)

    # Combine legends from both axes if needed
    handles1, labels1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Remove "_nolegend_" entries
        combined = [(h, l) for h, l in (list(zip(handles1, labels1)) +
                                        list(zip(handles2, labels2)))
                    if l != "_nolegend_"]
        if combined:
            handles, labels = zip(*combined)

            ax.legend(handles, labels, **legend_params)
    else:
        # Clean legend for single axis
        filtered = [(h, l) for h, l in zip(handles1, labels1) if l != "_nolegend_"]
        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(handles, labels, **legend_params)

    # Attach secondary axis for external use without breaking return type
    ax.independent_ax = ax2
    return ax

def plot_radar(ax, data, measures: list, by_condition=True, by_partner=True,
               ylim=None, item_labels=None, exclude = None, title=''):
    """Plots radar chart of trust survey responses for given data points.

    [by_condition, by_partner] = [True,False]: one line per condition
    [by_condition, by_partner] = [False,True]: one line per partner_type
    [by_condition, by_partner] = [True,True]: one line per (condition, partner_type) combo
    """
    legend_params = dict(frameon=False, loc='upper right',bbox_to_anchor=(1.3, 1.1),
                         ncol= 1#sum([by_condition,  by_partner])
                         )
    legend_params.update(**global_params['legend'])
    condition_col = "condition"
    partner_col = "partner_type"

    # Decide which columns define groups
    group_cols = []
    if by_condition:
        group_cols.append(condition_col)
    if by_partner:
        group_cols.append(partner_col)

    if not group_cols:
        raise ValueError("At least one of by_condition or by_partner must be True.")

    # Aggregate means for all requested measures
    grouped = data.groupby(group_cols)[measures].mean()

    # Group index (single index or MultiIndex)
    groups = list(grouped.index)
    n_groups = len(groups)
    n_measures = len(measures)

    if n_measures == 0 or n_groups == 0:
        return ax


    # Ensure polar axis
    if ax.name != "polar":
        # hide original ax
        ax.set_visible(False)

        # If caller passed a normal Axes, convert to polar
        # (matplotlib will create a new axes on the same figure)
        fig = ax.figure
        ax = fig.add_subplot(ax.get_subplotspec(), projection="polar",polar=True)
        print("WARNING!!!!!: Input ax was not polar; created new polar axes.")

    # Angles for each measure (one per variable)
    angles = np.linspace(0, 2 * np.pi, n_measures, endpoint=False)
    # Close the loop
    angles = np.concatenate([angles, angles[:1]])

    # Set radial limits based on data
    all_vals = grouped[measures].values.astype(float)
    rmin = np.nanmin(all_vals)
    rmax = np.nanmax(all_vals)
    if np.isnan(rmin) or np.isnan(rmax):
        rmin, rmax = 0.0, 1.0
    if rmax == rmin:
        pad = 1.0
    else:
        pad = 0.05 * (rmax - rmin)

    ylim = ylim or (0, rmax + pad)
    # ax.set_ylim(rmin - pad, rmax + pad)
    ax.set_ylim(*ylim)

    legend_handles = []
    legend_labels = []

    for key in groups:
        # Normalize key to tuple for consistent handling
        if not isinstance(key, tuple):
            key_tuple = (key,)
        else:
            key_tuple = key

        if exclude is not None:
            if isinstance(exclude, list):
                if any([k in exclude for k in key_tuple]):
                    continue
            elif isinstance(exclude, str) or isinstance(exclude, int):
                if exclude in key_tuple:
                    continue
            else:
                raise ValueError("exclude parameter must be a list, str, or int.")


        # Build label (match style from your grandmean function)
        label_parts = [
            f"C{val}" if val in [0, 1] else f"{val}"
            for col, val in zip(group_cols, key_tuple)
        ]
        label = ", ".join(label_parts)

        # Choose style based on condition/partner (reuse bar_style if available)
        style = {}
        try:
            if by_condition and by_partner:
                cond_val, partner_val = key_tuple
                style.update(line_style.get(cond_val, {}))
                style.update(line_style.get(partner_val, {}))
            elif by_condition and not by_partner:
                cond_val = key_tuple[0]
                style.update(line_style.get(cond_val, {}))
            elif by_partner and not by_condition:
                partner_val = key_tuple[0]
                style.update(line_style.get(partner_val, {}))
        except NameError:
            # bar_style not defined; fall back to defaults
            style = {}

        # Values for this group
        vals = grouped.loc[key, measures].values.astype(float)
        vals = np.concatenate([vals, vals[:1]])  # close the loop

        # Plot line
        line, = ax.plot(angles, vals, label=label, **style)

        # Fill area with light alpha (use style color if available)
        fill_color = style.get("color", line.get_color())
        fill_alpha = style.get("alpha", 0.1)
        ax.fill(angles, vals, color=fill_color, alpha=fill_alpha)

        legend_handles.append(line)
        legend_labels.append(label)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(measures if item_labels is None else item_labels)
    ax.set_title(title, y=1.1)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, **legend_params)

    return ax



def main():
    all_data = gather(flat=False)
    excluded_data = all_data.loc[all_data['include']==0]
    data = all_data.loc[all_data['include']==1]

    print(f'Data Summary:')
    print(f'\t| All data points: N={all_data["ID"].nunique()}')
    print(f"\t| Excluded N={excluded_data['ID'].nunique()} data points from analysis.")
    print(f"\t| Included N-{data['ID'].nunique()} data points in analysis.")

    plt.ioff()
    axh = 4
    axw = 5


    ###########################################
    ####### Metadata #########################
    sz = (2, 2)
    fig, axs = plt.subplots(*sz, figsize=(axw * sz[1], axh * sz[0]), constrained_layout=True)
    if np.ndim(axs) == 1: axs = axs[np.newaxis, :]
    # r=0
    # plot_params = {'groupby': ['partner_type', 'sex']}
    # plot_grandmean_bar2(axs[r, 0], data, measures=['dtrust'], ylabel="$\Delta Trust$",  title="Trust Calibration by Sex", **plot_params)
    # plot_grandmean_bar2(axs[r, 1], data, measures=['Reward'], ylabel="Reward", title="Reward by Sex", **plot_params)
    #
    r = 1
    plot_params = {'groupby': ['partner_type','layout']}
    plot_grandmean_bar2(axs[r, 0], data, measures=['dtrust'], ylabel="$\Delta Trust$", title="Trust Calibration by Layout", **plot_params)
    plot_grandmean_bar2(axs[r, 1], data, measures=['Reward'], ylabel="Reward", title="Reward by Layout", **plot_params)


    ###########################################
    ####### TIMESERIES PLOTS #########################
    sz = (2, 3)
    r=0
    fig, axs = plt.subplots(*sz, figsize=(axw * sz[1], axh * sz[0]), constrained_layout=True)
    if np.ndim(axs) == 1: axs = axs[np.newaxis, :]
    plot_timeseries(axs[r, 0], data, measure='trust_score', scatter=False, title=f"Timeseries Trust (Partner x Cond)",ylabel="Trust Score")
    plot_timeseries(axs[r, 1], data, measure='Reward', scatter=False, title=f"Timeseries Reward (Partner x Cond)",ylabel="Reward")

    r=1
    plot_timeseries(axs[r, 0], data, measure='C-ACT', scatter=False, title=f"Timeseries C-ACT (Partner x Cond)", ylabel="C-ACT")
    plot_timeseries(axs[r, 1], data, measure='H-IDLE', scatter=False, title=f"Timeseries H-IDLE (Partner x Cond)", ylabel="H-IDLE")
    plot_timeseries(axs[r, 2], data, measure='R-IDLE', scatter=False, title=f"Timeseries R-IDLE (Partner x Cond)",  ylabel="R-IDLE")


    ###########################################
    ####### GAME RESULTS ######################
    # sz = (2,3)
    # fig, axs = plt.subplots(*sz,figsize=(axw*sz[1],axh*sz[0]), constrained_layout=True)
    # if np.ndim(axs) == 1: axs = axs[np.newaxis, :]
    # # ----------------------------------------
    # for r in range(sz[0]):
    #     plot_params = {'by_condition': bool(r), 'by_partner': True}
    #     tlabel = '(by'
    #
    #     if plot_params['by_partner']:
    #         tlabel += ' Partner'
    #     if plot_params['by_condition'] and plot_params['by_partner']:
    #         tlabel += ' X'
    #     if plot_params['by_condition']:
    #         tlabel += ' Condition'
    #     tlabel += ')'
    #
    #     if r ==0:
    #         plot_timeseries(axs[r,0], data, measure='trust_score', scatter=False,
    #                         title=f"Trust Score Over Games {tlabel}", ylabel="Trust Score")
    #     else:
    #         plot_grandmean_bar2(axs[r, 0], data, measures=['dtrust'], ylabel="$\Delta Trust$",
    #                             title="Trust Calibration by Sex", groupby = ['partner_type', 'sex'])
    #
    #     plot_grandmean_bar(axs[r,1], data,
    #                        # measures=['dtrust','trust_slope'],
    #                        measures=['dtrust'],
    #                        ylabel="$\Delta Trust$", title=f"Mean Trust Calibration {tlabel}",
    #                        zero_line=True,
    #                        **plot_params)
    #
    #     plot_grandmean_bar(axs[r,2], data, measures=['Reward','C-ACT','H-IDLE','R-IDLE','H-Pred'],
    #                        independent_scale=['C-ACT','H-IDLE','R-IDLE','H-Pred'],
    #                        ylabel="Reward", title = f"Other Measures {tlabel}", **plot_params)

    # ----------------------------------------
    # r += 1
    # plot_params = {'by_condition': True, 'by_partner': True}
    # plot_timeseries(axs[r, 0], data, measure='trust_score', scatter=True)
    # plot_grandmean_bar(axs[r, 1], data, measures=['dtrust', 'trust_slope'],
    #                    ylabel="$\Delta Trust$", title="Mean Trust Calibration", **plot_params)
    # plot_grandmean_bar(axs[r, 2], data, measures=['Reward', 'C-ACT', 'H-IDLE', 'R-IDLE'],
    #                    independent_scale=['C-ACT', 'H-IDLE', 'R-IDLE'],
    #                    ylabel="Reward", title="Other Measures", **plot_params)

    ###########################################
    ####### TRUST SURVEY RADAR PLOTS ##########
    # sz = (1, 3)
    # plot_params = {'measures':DataPoint.get_trust_questions(df_labels=True),
    #               'item_labels':DataPoint.get_trust_questions(df_labels=False),
    #               'by_condition':False, 'by_partner':True, 'ylim': (0,1)}
    #
    # fig, axs = plt.subplots(*sz, figsize=(axw * sz[1], axh * sz[0]), constrained_layout=True)
    # if np.ndim(axs) == 1: axs = axs[np.newaxis, :]
    #
    # r = 0
    # plot_radar(axs[r, 0], data, title='Trust Survey Responses', **plot_params)
    # plot_radar(axs[r, 1], data, exclude=1, title='Cond 0 Trust Survey Responses', **plot_params)
    # plot_radar(axs[r, 2], data, exclude=0, title='Cond 1 Trust Survey Responses', **plot_params)


    plt.show()

if __name__ == "__main__":
    main()