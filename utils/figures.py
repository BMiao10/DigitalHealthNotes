import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import seaborn as sns

import regex as re

def mask_small_group(demographics, min_n=10):
    """
    Masks counts of groups with less than min_n values and excludes them from proportion calculations
    
    """
    mask_values = demographics[demographics["count"]<min_n]
    mask_values["count"] = "<10"
    mask_values["proportion"] = np.nan
    
    # get masked proportions
    all_values = demographics[demographics["count"]>=min_n]
    all_values["proportion"] = all_values["count"] / demographics["count"].sum()
    
    return pd.concat([all_values, mask_values])

def retrieve_dh_sentences(notes, term, extend=True):
    """
    Retrieves the sentences that contain the digital health terms.
    Also retrieves the previous and next sentences.

    Parameters
    ----------
    note_text : str
        The text of the note.
    dh_terms : list
        The list of digital health terms.
    extend : bool
        Whether to include former and previous sentences
    
    Returns
    -------
    list
        The list of sentences that contain the digital health terms.
    """
    dh_sentences = []
    regexp = re.compile(term)

    if extend: # get DH sentence and sentence before/after
        for curr_note in notes:
            curr_sent = []
            
            for i, sentence in enumerate(curr_note):
                if regexp.search(sentence):
                    curr_sent.append(sentence)

                    if i > 0:
                        curr_sent.append(curr_note[i-1])
                    if i < len(curr_note)-1:
                        curr_sent.append(curr_note[i+1])
                
            dh_sentences.append(list(set(curr_sent)))

        return dh_sentences
    else:
        for curr_note in notes:
            curr_sent = [sentence for sentence in curr_note if regexp.search(sentence)]
            dh_sentences.append(curr_sent)
    
    return dh_sentences

def plot_notes_over_time(notes_df, time_col="year", hue="encounter_department_specialty", # masked name
                         hue_label="Department specialty", count="encounterkey", 
                         top_n=10, add_total=False, **kwargs):
    """
    Plots number of notes over time based on a hue
    
    Parameters:
    ---------
    notes_df: pd.DataFrame
        DataFrame containing values to plot
    hue: str
        The column name of the hue you want to plot.
    hue_label: str
        The label you want to give the hue.
    count: str
        The column name of the count you want to plot.
    top_n: int
        The number of top values you want to plot.
    TODO: add_total: bool
        Whether to add a line for overall values
    
    Returns:
    --------
    ax: matplotlib.axes._subplots.AxesSubplot
        The plot.
    
    Example
    --------
    notes_over_time(hue="encounter_department_specialty", hue_label="Department specialty", count="encounterkey", top_n=10)
    
    """
    values_df = notes_df.groupby([time_col, hue])[count].count().reset_index()
    values_df = values_df.sort_values(count, ascending=False)
    values_df.columns = ["Year", hue_label, "Count"]

    # Add totals
    total_df = notes_df.groupby([time_col])[count].count().reset_index()
    total_df[hue_label] = "Total"
    total_df.columns = ["Year", "Count", hue_label]
    total_df = total_df[["Year", hue_label, "Count"]]
    plot_df = pd.concat([total_df, values_df])

    ## Plot counts for top categories
    top_values = list(notes_df.value_counts(hue).sort_values(ascending=False)[:top_n].index)
    plot_df = plot_df[plot_df[hue_label].isin(top_values)]

    fig, ax = plt.subplots(figsize=(12,8))
    ax = sns.lineplot(data=plot_df, x="Year", y="Count", hue=hue_label, **kwargs)
    ax.set(yscale='log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    return ax

def calculate_cagr(values_df, group_col, time_col="Year", count_col="Count"):
    """
    Calculate the CAGR for a pandas DataFrame with columns for Year (or other time)
    
    Parameters:
    ---------
    values_df: pd.DataFrame
        Dataframe containing categories, time, and count
    group_col : str
        The name of the column containing the categories to calculate CAGR for.
    time_col : str
        The name of the column containing the time period.
    count_col : str
        The name of the column containing the count.
    
    Returns
    -------
    cagr_df : pandas DataFrame
        A DataFrame with the group, CAGR, and count.
        
    Example
    -------
    notes_cagr(group_col="Department specialty", time_col="Year", count_col="Count")
    """
    xtab_df = pd.crosstab(index=values_df[time_col], columns=values_df[group_col], 
                      values=values_df[count_col], aggfunc=np.sum, normalize=False)
    counts = xtab_df.sum(axis=0) # get counts before changing values
    xtab_df = xtab_df.replace(np.nan,1)
    xtab_df = xtab_df.replace(0,1)

    # Calculate compound annual growth rate (CAGR)
    cagr_df = xtab_df.pct_change().add(1).prod().pow(1./(len(xtab_df.index) - 1)).sub(1)*100
    cagr_df.sort_values()[-20:]

    cagr_df = cagr_df.reset_index()
    cagr_df["Count"] = cagr_df[group_col].map(counts)
    cagr_df["Count"] = cagr_df["Count"].astype(int)
    cagr_df.columns = [group_col, "CAGR (%)", "Count"]
    cagr_df = cagr_df.set_index(group_col).sort_values("CAGR (%)")
    
    return cagr_df


def ridge_plot(plot_df, hue, order, pal, vmin=None, vmax=None):
    """
    Ridge plot of density over time
    
    Parameters
    ----------
    plot_df : pd.DataFrame
        contains columns for hue (row values), "CAGR", "Year", 
    hue : str
        The name of the column in the dataframe to be used as the hue (row values).
    order : list
        The order in which the categories should be plotted.
    pal : seaborn color palette
        The color palette to be used for the plots.
    
    Returns
    -------
    sns.FacetGrid
        The function creates the ridge plots and displays them.
    """
    
    # Get colors
    vmin = plot_df['CAGR'].min() if vmin is None else vmin
    vmax = plot_df['CAGR'].max() if vmin is None else vmax
    normalize = plt.Normalize(vmin=vmin, vmax=vmax)

    plot_df["color"] = [c for c in pal(normalize(plot_df['CAGR']))]
    plot_df["color_hex"]= [mcolors.rgb2hex(c) for c in plot_df["color"] ]
    hue_to_color = dict(zip(plot_df[hue], plot_df["color_hex"], ))
    discrete_pal = sns.color_palette([hue_to_color[o] for o in order])

    # Create FacetGrid
    g = sns.FacetGrid(plot_df, row=hue, hue=hue, aspect=8.2, row_order=order,
                      height=0.75, palette=discrete_pal, hue_order=order) #,  #, palette=pal

    # Draw the densities in a few steps
    kdeargs = {"weights":"Count", "clip":[2011,2022],#"hue": "CAGR", 
               "palette":plot_df["color"], "hue_order": order}

    g.map_dataframe(sns.kdeplot, "Year", bw_adjust=.4, clip_on=False,
                    fill=True, alpha=0.9, linewidth=1.5, **kdeargs)
    g.map_dataframe(sns.kdeplot, "Year", clip_on=False, color="w", 
                    lw=2, bw_adjust=.4, **kdeargs)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.45, label, color=color, ha="left", va="top", transform=ax.transAxes) #

    g.map(label, hue)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.22)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add colormap
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=pal)
    scalarmappaple.set_array([])

    cax = g.figure.add_axes([1.01, 0.58, 0.05, 0.3])
    cbar = g.figure.colorbar(scalarmappaple, cax = cax)
    cbar.set_label("CAGR (%)", fontsize=12)
    
    return g


    