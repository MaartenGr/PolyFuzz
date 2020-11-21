import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


def precision_recall_curve(model, precision_steps=0.01):
    """ Calculate precision recall curve based on minimum similarity between strings """
    min_precisions = np.arange(0., 1 + precision_steps, precision_steps)
    average_precision = []
    recall = []
    similarities = model.Similarity.values
    total = len(model)

    for min_precision in min_precisions:
        selection = similarities[similarities >= min_precision]
        recall.append(sum(selection) / total)
        average_precision.append(np.mean(selection))

    return min_precisions, recall, average_precision


def visualize_precision_recall(matches, min_precision, recall):
    if not isinstance(matches, dict):
        matches = {"Model": matches}
        min_precision = {"Model": min_precision}
        recall = {"Model": recall}

    # Create single dataset of similarity score for all models
    distribution_data = [(matches[name].Similarity.values, [name for _ in range(len(matches[name]))]) for name in
                         matches.keys()]
    distribution_data = pd.DataFrame(np.hstack(distribution_data).T, columns=["Similarity", "Model"])
    distribution_data.Similarity = distribution_data.Similarity.astype(float)
    model_names = list(matches.keys())

    # Create layout
    cmap = get_cmap('Accent')
    fig = plt.figure(figsize=(20, 4))
    widths = [3, .1, 1.5]
    heights = [1]
    gs = gridspec.GridSpec(1, 3, width_ratios=widths, height_ratios=heights)
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[:, 2], sharex=ax1)

    # Precision-recall curve
    for color, model_name in zip(cmap.colors, model_names):
        ax1.plot(min_precision[model_name], recall[model_name], color=color)

    # Similarity Histogram
    for color, model_name in zip(cmap.colors, model_names):
        sns.kdeplot(matches[model_name]["Similarity"], fill=True, ax=ax2, color=color)
    ax2.yaxis.tick_right()
    ax2.set_xlabel("Similarity Score")
    ax2.set_xlim(left=0, right=1)

    # Titles
    if len(model_names) == 1:
        fig.suptitle(f'Score Metrics', size=20, y=1.1, x=0.5)
    else:
        fig.suptitle('Score Metrics', size=20, y=1.1, x=0.5)
    plt.setp([ax1], title='Effect of Minimum Similarity')
    plt.setp([ax2], title='Score Frequency')

    # Custom Legend
    if len(model_names) != 1:
        custom_lines = [Line2D([0], [0], color=color, lw=4) for color, model_name in zip(cmap.colors, model_names)]
        ax1.legend(custom_lines, model_names, bbox_to_anchor=(1.01, .7, 1., .102), loc=3,
                   ncol=1, borderaxespad=0., frameon=False)
