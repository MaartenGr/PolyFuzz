import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


def precision_recall_curve(matches: pd.DataFrame,
                           precision_steps: float = 0.01) -> Tuple[np.ndarray,
                                                                   List[float],
                                                                   List[float]]:
    """ Calculate precision recall curve based on minimum similarity between strings

    A minimum similarity score might be used to identify
    when a match could be considered to be correct. For example,
    we can assume that if a similarity score pass 0.95 we are
    quite confident that the matches are correct. This minimum
    similarity score can be defined as **precision** since it shows
    you how precise we believe the matches are at a minimum.

    **Recall** can then be defined as as the percentage of matches
    found at a certain minimum similarity score. A high recall means
    that for a certain minimum precision score, we find many matches.

    Arguments:
        matches: contains the columns *From*, *To*, and *Similarity* used for calculating
                 precision, recall, and average precision
        precision_steps: the incremental steps in minimum precision

    Returns:
        min_precisions: minimum precision steps
        recall: recall per minimum precision step
        average_precision: average precision per minimum precision step
    """
    min_precisions = list(np.arange(0., 1 + precision_steps, precision_steps))
    average_precision = []
    recall = []
    similarities = matches.Similarity.values
    total = len(matches)

    for min_precision in min_precisions:
        selection = similarities[similarities >= min_precision]
        recall.append(len(selection) / total)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_precision.append(float(np.mean(selection)))

    return min_precisions, recall, average_precision


def visualize_precision_recall(matches, min_precisions, recall):
    """ Visualize the precision recall curve for one or more models

    Arguments:
        matches: contains the columns *From*, *To*, and *Similarity* used for calculating
                 precision, recall, and average precision
        min_precisions: minimum precision steps
        recall: recall per minimum precision step
    """
    if not isinstance(matches, dict):
        matches = {"Model": matches}
        min_precisions = {"Model": min_precisions}
        recall = {"Model": recall}

    # Create single dataset of similarity score for all models
    distribution_data = [(matches[name].Similarity.values, [name for _ in range(len(matches[name]))]) for name in
                         matches.keys()]
    distribution_data = pd.DataFrame(np.hstack(distribution_data).T, columns=["Similarity", "Model"])
    distribution_data.Similarity = distribution_data.Similarity.astype(float)
    model_names = list(matches.keys())

    # Create layout
    cmap = get_cmap('Accent')
    fig = plt.figure(figsize=(20, 5))
    widths = [1.5, .1, 1.5]
    heights = [1.5]
    gs = gridspec.GridSpec(1, 3, width_ratios=widths, height_ratios=heights)
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[:, 2], sharex=ax1)

    # Precision-recall curve
    for color, model_name in zip(cmap.colors, model_names):
        ax1.plot(min_precisions[model_name], recall[model_name], color=color)
    ax1.set_ylim(bottom=0, top=1)
    ax1.set_xlim(left=0, right=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel(r"$\bf{Precision}$" + "\n(Minimum Similarity)")
    ax1.set_ylabel(r"$\bf{Recall}$" + "\n(Percentage Matched)")

    # Similarity Histogram
    for color, model_name in zip(cmap.colors, model_names):
        sns.kdeplot(matches[model_name]["Similarity"], fill=True, ax=ax2, color=color)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlabel(r"$\bf{Similarity}$")
    ax2.set_ylabel("")
    ax2.set_xlim(left=-0, right=1)

    # Titles
    if len(model_names) == 1:
        fig.suptitle(f'Score Metrics', size=20, y=1, x=0.5)
    else:
        fig.suptitle('Score Metrics', size=20, y=1, x=0.5)
    plt.setp([ax1], title='Precision-Recall Curve')
    plt.setp([ax2], title='Score Frequency - KDE')

    # Custom Legend
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color, model_name in zip(cmap.colors, model_names)]
    ax1.legend(custom_lines, model_names, bbox_to_anchor=(1.05, .61, .7, .902), loc=3,
               ncol=1, borderaxespad=0., frameon=True)
