import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Mapping
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


def precision_recall_curve(matches: pd.DataFrame,
                           precision_steps: float = 0.01) -> Tuple[List[float],
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


def visualize_precision_recall(matches: Mapping[str, pd.DataFrame],
                               min_precisions: Mapping[str, List[float]],
                               recall: Mapping[str, List[float]],
                               kde: bool = True,
                               save_path: str = None):
    """ Visualize the precision recall curve for one or more models

    Arguments:
        matches: contains the columns *From*, *To*, and *Similarity* used for calculating
                 precision, recall, and average precision per model
        min_precisions: minimum precision steps per model
        recall: recall per minimum precision step per model
        kde: whether to also visualize the kde plot
        save_path: the path to save the resulting image to

    Usage:

    ```python
    visualize_precision_recall(matches, min_precisions, recall, save_path="data/results.png")
    ```
    """
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

    if len(model_names) == 1:
        middle = 0
    else:
        middle = .1

    if kde:
        widths = [1.5, middle, 1.5]
    else:
        widths = [1.5, middle, 0]

    heights = [1.5]
    gs = gridspec.GridSpec(1, 3, width_ratios=widths, height_ratios=heights)
    ax1 = plt.subplot(gs[:, 0])

    if kde:
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
    if kde:
        for color, model_name in zip(cmap.colors, model_names):
            sns.kdeplot(matches[model_name]["Similarity"], fill=True, ax=ax2, color=color)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_xlabel(r"$\bf{Similarity}$")
        ax2.set_ylabel("")
        ax2.set_xlim(left=-0, right=1)
        plt.setp([ax2], title='Score Frequency - KDE')

    # Titles
    if len(model_names) == 1 and kde:
        fig.suptitle(f'Score Metrics', size=20, y=1, x=0.5)
        plt.setp([ax1], title='Precision-Recall Curve')
    elif kde:
        fig.suptitle('Score Metrics', size=20, y=1, x=0.5)
        plt.setp([ax1], title='Precision-Recall Curve')
    else:
        fig.suptitle('Precision-Recall Curve', size=20, y=1, x=0.45)

    # Custom Legend
    if len(model_names) > 1:
        custom_lines = [Line2D([0], [0], color=color, lw=4) for color, model_name in zip(cmap.colors, model_names)]
        ax1.legend(custom_lines, model_names, bbox_to_anchor=(1.05, .61, .7, .902), loc=3,
                   ncol=1, borderaxespad=0., frameon=True, fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300)
