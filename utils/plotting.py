import matplotlib.pyplot as plt

def annotate_superordinates(ax, superordinate_positions, unique_superordinates):
    """
    Add superordinate annotations to the plot.
    """
    ax.set_xticks(superordinate_positions)
    ax.set_yticks(superordinate_positions)
    ax.set_xticklabels(unique_superordinates, rotation=90, fontsize=12, ha='center')
    ax.set_yticklabels(unique_superordinates, fontsize=12, va='center')

def get_annotation_positions(sorted_superordinates, sorted_labels):
    """
    Helper function to get positions for superordinate and class annotations.
    """
    unique_superordinates = []
    superordinate_positions = []
    class_positions = []
    current_superordinate = None
    superordinate_start = 0
    current_class = None
    class_start = 0

    for i, (superordinate, label) in enumerate(zip(sorted_superordinates, sorted_labels)):
        if superordinate != current_superordinate:
            if current_superordinate is not None:
                class_positions.append(((class_start + i - 1) / 2, current_class))
                unique_superordinates.append(current_superordinate)
                superordinate_positions.append((superordinate_start + i) / 2)
            current_superordinate = superordinate
            superordinate_start = i
            current_class = None

        if label != current_class:
            if current_class is not None:
                class_positions.append(((class_start + i) / 2, current_class))
            current_class = label
            class_start = i

    # Add the last superordinate and its last class
    unique_superordinates.append(current_superordinate)
    superordinate_positions.append((superordinate_start + len(sorted_superordinates)) / 2)
    class_positions.append(((class_start + len(sorted_labels)) / 2, current_class))

    return superordinate_positions, unique_superordinates, class_positions

def annotate_classes(ax, class_positions):
    """
    Add class label annotations to the plot.
    """
    ax2 = ax.twiny()
    ax3 = ax.twinx()

    class_positions_values = [pos for pos, _ in class_positions]
    class_labels = [label for _, label in class_positions]

    ax2.set_xticks(class_positions_values)
    ax2.set_xticklabels(class_labels, rotation=90, fontsize=6, ha='left')
    ax2.set_xlim(ax.get_xlim())

    ax3.set_yticks(class_positions_values)
    ax3.set_yticklabels(class_labels, fontsize=6, va='bottom')
    ax3.set_ylim(ax.get_ylim())

def add_class_lines(ax, class_positions):
    """
    Add lines to group classes.
    """
    for pos, _ in class_positions[1:]:
        ax.axhline(y=pos, color='gray', linestyle=':', linewidth=0.5)
        ax.axvline(x=pos, color='gray', linestyle=':', linewidth=0.5)

def add_superordinate_lines(ax, superordinate_positions):
    """
    Add lines to group superordinates.
    """
    for i in range(1, len(superordinate_positions)):
        pos = (superordinate_positions[i] + superordinate_positions[i-1]) / 2
        ax.axhline(y=pos, color='white', linestyle='--', linewidth=1)
        ax.axvline(x=pos, color='white', linestyle='--', linewidth=1)

def add_percentage_labels(bars):
    """
    Add percentage labels on top of bars.
    """
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12)

def add_statistical_summary(plt, total_activations, zero_activations, nonzero_activations, raw_activations):
    """
    Add statistical summary to the plot.
    """
    summary_stats = {
        'Total Activations': f"{total_activations:,}",
        'Zero Activations': f"{zero_activations:,}",
        'Non-zero Activations': f"{nonzero_activations:,}",
        'Mean (non-zero)': f"{np.mean(raw_activations[raw_activations != 0]):.3f}",
        'Std (non-zero)': f"{np.std(raw_activations[raw_activations != 0]):.3f}"
    }
    
    stats_text = '\n'.join([f'{k}: {v}' for k, v in summary_stats.items()])
    plt.text(0.98, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8,
                    edgecolor='gray', boxstyle='round,pad=0.5'))
