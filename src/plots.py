import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Custom color palette
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
text_color = '#FFFFFF'  # White text


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Noto Sans CJK JP']


def load_log_data(file_path):
    """
    Load JSON-formatted log data from a file.

    Args:
    file_path (str): Path to the log file.

    Returns:
    list: A list of dictionaries, where each dictionary represents a log entry.

    This function reads a JSON-formatted log file and returns its contents as a list of Python dictionaries.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def load_id_to_char_map(json_file_path):
    """
    Load and create a mapping from ID to character from a JSON file.

    Args:
    json_file_path (str): Path to the JSON file containing ID to character mappings.

    Returns:
    dict: A dictionary mapping numeric IDs to their corresponding characters.

    This function reads a JSON file containing character mappings and creates a reverse mapping
    where the keys are numeric IDs and the values are the corresponding characters.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create a reverse mapping: id to character
    id_to_char = {v: k.split()[0] for k, v in data['codepoint_to_id'].items()}
    return id_to_char


def setup_plot_style():
    """
    Set up the global plotting style for matplotlib and seaborn.

    This function configures various plot parameters including:
    - Dark background theme
    - Grid properties
    - Text colors
    - Figure and axes colors
    - Spacing and padding

    The style is optimized for readability and aesthetic appeal on dark backgrounds.
    """
    plt.style.use('dark_background')

    # Set up the basic style
    sns.set_style("darkgrid", {
        "axes.facecolor": "#1C1C1C",
        "grid.color": "#303030",
    })

    # Adjust grid properties
    plt.rcParams['grid.color'] = '#303030'  # Darker gray for grid lines
    plt.rcParams['grid.linewidth'] = 0.5  # Thinner grid lines
    plt.rcParams['axes.facecolor'] = '#000000'  # Pure black for contrast
    plt.rcParams['axes.edgecolor'] = '#3C3C3C'  # Subtle edge color for axes
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True  # Ensure grid is drawn behind the lines

    # Set text colors
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    # Additional settings to ensure text visibility
    plt.rcParams['figure.facecolor'] = '#000000'
    plt.rcParams['savefig.facecolor'] = '#000000'

    # Adjust spacing
    plt.rcParams['figure.autolayout'] = False  # Disable automatic tight layout
    plt.rcParams['figure.subplot.top'] = 0.88  # More space at the top for the title
    plt.rcParams['figure.subplot.bottom'] = 0.11  # More space at the bottom for x-label
    plt.rcParams['figure.subplot.left'] = 0.08  # More space on the left for y-label
    plt.rcParams['figure.subplot.right'] = 0.92  # Keep some space on the right

    # Increase padding between plot elements
    plt.rcParams['axes.titlepad'] = 12  # Space between title and plot
    plt.rcParams['axes.labelpad'] = 8  # Space between axis labels and tick labels


def create_custom_cmap(name, colors):
    """
    Create a custom color map for plots.

    Args:
    name (str): Name of the custom colormap.
    colors (list): List of colors to use in the colormap.

    Returns:
    matplotlib.colors.LinearSegmentedColormap: A custom colormap object.

    This function creates a linear segmented colormap from the given list of colors,
    which can be used in various plotting functions for customized color schemes.
    """
    return LinearSegmentedColormap.from_list(name, colors)


def plot_metrics(data, metrics, title, filename):
    """
    Plot multiple metrics over epochs.

    Args:
    data (list): List of dictionaries containing epoch data.
    metrics (list): List of metric names to plot.
    title (str): Title of the plot.
    filename (str): Path to save the output image.

    This function creates a line plot for each specified metric over epochs.
    Different metrics are plotted in different colors on the same graph.
    """
    epochs = [entry['epoch'] for entry in data]

    fig, ax = plt.subplots(figsize=(12, 6))
    for metric in metrics:
        values = [entry[metric] for entry in data]
        ax.plot(epochs, values, label=metric.replace('_', ' ').title(), linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()


def plot_top_misclassifications(data, epoch, id_to_char, top_n=50, filename='../plots/top_misclassifications.png'):
    """
    Plot the top misclassifications for a given epoch.

    Args:
    data (list): List of dictionaries containing epoch data.
    epoch (int): The epoch number to plot misclassifications for.
    id_to_char (dict): Mapping of label IDs to their corresponding characters.
    top_n (int, optional): Number of top misclassifications to plot. Defaults to 50.
    filename (str, optional): Path to save the output image.

    This function creates a horizontal bar plot showing the top misclassifications.
    Each bar represents a misclassification pair (true label → predicted label),
    with the length of the bar indicating the count of misclassifications.
    """
    conf_matrix = data[epoch - 1]['val_confusion_matrix']
    misclassifications = []
    for true_label in conf_matrix:
        for pred_label, count in conf_matrix[true_label].items():
            misclassifications.append((int(true_label), int(pred_label), count))

    top_misc = sorted(misclassifications, key=lambda x: x[2], reverse=True)[:top_n]

    # Increase figure height to accommodate more space between labels
    fig, ax = plt.subplots(figsize=(12, top_n * 0.4))  # Adjust the multiplier (0.4) as needed
    y_pos = np.arange(len(top_misc))
    counts = [m[2] for m in top_misc]
    labels = [f'{id_to_char[m[0]]} → {id_to_char[m[1]]}' for m in top_misc]

    # Remove white borders by setting edge color to None
    ax.barh(y_pos, counts, align='center', color=colors, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14, fontname='Hiragino Sans GB')
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Misclassifications (Epoch {epoch})', fontsize=16, fontweight='bold')

    # Add more padding to the left to ensure labels are not cut off
    plt.subplots_adjust(left=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()
