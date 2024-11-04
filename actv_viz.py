import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy

from models import CNNAnalyzer, ViTAnalyzer

def load_class_info():
    df = pd.read_csv('categories/ranked_Imagenet.csv')
    wnid_to_description = dict(zip(df['wnid'], df['description']))
    return wnid_to_description

def load_superordinate_info():
    """
    Returns:
        superordinate_dict (dict): dictionary of superordinate categories to WNIDs
        all_wnids (set): set of all WNIDs in the superordinate categories
    """
    superordinate_dict = {}
    all_wnids = set()
    for superordinate in unique_superordinates:
        df_super = pd.read_csv(f'categories/{superordinate}_Imagenet.csv')
        wnids = set(df_super['wnid'].values[:num_classes_per_superordinate])
        superordinate_dict[superordinate] = wnids
        all_wnids.update(wnids)
    return superordinate_dict, all_wnids

def get_raw_activations(all_wnids, layer_name):
    """
    Helper function to get raw activations for a given layer from saved files.
    
    Args:
        layer_name (str): Name of the layer
    
    Returns:
        np.ndarray: Array of activations for all images
    """
    all_activations = []
    for wnid in all_wnids:
        print(f"Processing raw activations for {wnid}")
        activation_dir = os.path.join(actv_output_dir, str(layer_name), wnid)
        print(f"  activation_dir: {activation_dir}")
        if not os.path.isdir(activation_dir):
            continue
            
        for activation_file in os.listdir(activation_dir):
            print(f"  activation_file: {activation_file}")
            if activation_file.endswith('.npy'):
                activation_path = os.path.join(activation_dir, activation_file)
                activation = np.load(activation_path)
                all_activations.append(activation)
    
    return np.array(all_activations)
    
def plot_2d_activations(activations_2d, superordinates, layer_name, output_dir):
    """
    Plot 2D activations colored by superordinate category.
    """
    plt.figure(figsize=(20, 20))
    color_map = matplotlib.colormaps["tab20"]
    colors = {s: color_map(i/len(unique_superordinates)) for i, s in enumerate(unique_superordinates)}
    
    for superordinate in unique_superordinates:
        mask = np.array(superordinates) == superordinate
        plt.scatter(activations_2d[mask, 0], 
                    activations_2d[mask, 1], 
                    c=[colors[superordinate]], 
                    label=superordinate, 
                    s=50,
                    alpha=0.7
                    )
    
    plt.title(f'{base_model_name} Activations - {layer_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_model_name}_activations_{layer_name}.png'), dpi=150)
    plt.close()

def plot_dissimilarity_matrix(dissimilarity_matrix, labels, superordinates, layer_name, base_model_name, dissimilarity_metric, output_dir):
    """
    Plot dissimilarity matrix with superordinate and class annotations.
    """
    fig, ax = plt.subplots(figsize=(30, 25))
    
    # Sort the dissimilarity matrix and labels based on superordinates and then by class labels
    sorted_indices = np.lexsort((labels, superordinates))
    sorted_dissimilarity_matrix = dissimilarity_matrix[sorted_indices][:, sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_superordinates = [superordinates[i] for i in sorted_indices]

    # Create heatmap
    im = ax.imshow(sorted_dissimilarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='dissimilarity')

    # Get superordinate and class positions
    superordinate_positions, unique_superordinates, class_positions = get_annotation_positions(
        sorted_superordinates, sorted_labels)

    # Annotate superordinates
    annotate_superordinates(ax, superordinate_positions, unique_superordinates)

    # Add superordinate group lines
    add_superordinate_lines(ax, superordinate_positions)

    # Annotate class labels
    annotate_classes(ax, class_positions)

    # Add class group lines
    add_class_lines(ax, class_positions)

    plt.title(f'dissimilarity Matrix ({dissimilarity_metric}) - {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 
        f'{base_model_name}_{dissimilarity_metric}_dissimilarity_matrix_{layer_name}.png'), 
        dpi=300)
    plt.close()

def plot_activation_distribution(raw_activations, layer_name, output_dir):
    """
    Plot activation distribution analysis including zero vs non-zero statistics.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics
    total_activations = raw_activations.size
    zero_activations = np.sum(raw_activations == 0)
    nonzero_activations = total_activations - zero_activations
    
    zero_percentage = (zero_activations / total_activations) * 100
    nonzero_percentage = (nonzero_activations / total_activations) * 100
    
    # Create bar plot
    bars = plt.bar(['Zero Activations', 'Non-zero Activations'], 
                [zero_percentage, nonzero_percentage],
                color=['#ff9999', '#66b3ff'])
    
    # Add percentage labels
    add_percentage_labels(bars)
    
    # Add title and labels
    plt.title(f'Activation Distribution Analysis - {layer_name}', fontsize=14, pad=20)
    plt.ylabel('Percentage of Total Activations (%)', fontsize=12)
    
    # Add statistical summary
    add_statistical_summary(plt, total_activations, zero_activations, 
                          nonzero_activations, raw_activations)
    
    # Customize the plot
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(zero_percentage, nonzero_percentage) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_model_name}_activation_distribution_{layer_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

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

def annotate_superordinates(ax, superordinate_positions, unique_superordinates):
    """
    Add superordinate annotations to the plot.
    """
    ax.set_xticks(superordinate_positions)
    ax.set_yticks(superordinate_positions)
    ax.set_xticklabels(unique_superordinates, rotation=90, fontsize=12, ha='center')
    ax.set_yticklabels(unique_superordinates, fontsize=12, va='center')

def add_superordinate_lines(ax, superordinate_positions):
    """
    Add lines to group superordinates.
    """
    for i in range(1, len(superordinate_positions)):
        pos = (superordinate_positions[i] + superordinate_positions[i-1]) / 2
        ax.axhline(y=pos, color='white', linestyle='--', linewidth=1)
        ax.axvline(x=pos, color='white', linestyle='--', linewidth=1)

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

def analyze_superordinate_dissimilarities(dissimilarity_matrix, superordinates, labels):
    """
    Analyze dissimilarities within and between superordinates.
    
    Args:
        dissimilarity_matrix (np.ndarray): The dissimilarity matrix
        superordinates (list): List of superordinate categories for each sample
        labels (list): List of class labels for each sample
        
    Returns:
        dict: Dictionary containing analysis results
    """
    n_superordinates = len(unique_superordinates)
    
    # Create masks for each superordinate
    superordinate_masks = {
        sup: np.array([s == sup for s in superordinates])
        for sup in unique_superordinates
    }
    
    # Calculate within-superordinate dissimilarities
    within_dissimilarities = {}
    within_dissimilarities_distributions = {}
    for sup in unique_superordinates:
        mask = superordinate_masks[sup]
        # Get upper triangle of dissimilarity matrix for this superordinate
        within_sim = dissimilarity_matrix[np.ix_(mask, mask)]
        # Extract upper triangle (excluding diagonal)
        upper_tri = within_sim[np.triu_indices_from(within_sim, k=1)]
        within_dissimilarities[sup] = np.mean(upper_tri)
        within_dissimilarities_distributions[sup] = upper_tri
    
    # Calculate between-superordinate dissimilarities
    between_dissimilarities = {}
    between_dissimilarities_distributions = {}
    for i, sup1 in enumerate(unique_superordinates):
        for j, sup2 in enumerate(unique_superordinates[i+1:], i+1):
            mask1 = superordinate_masks[sup1]
            mask2 = superordinate_masks[sup2]
            between_sim = dissimilarity_matrix[np.ix_(mask1, mask2)]
            between_dissimilarities[(sup1, sup2)] = np.mean(between_sim)
            between_dissimilarities_distributions[(sup1, sup2)] = between_sim.flatten()
    
    # Perform statistical tests
    statistical_tests = {}
    
    # 1. ANOVA test for differences among within-superordinate dissimilarities
    within_groups = list(within_dissimilarities_distributions.values())
    f_stat, p_value = scipy.stats.f_oneway(*within_groups)
    statistical_tests['within_anova'] = {
        'f_statistic': f_stat,
        'p_value': p_value
    }
    
    # 2. Post-hoc t-tests for within-superordinate comparisons (with Bonferroni correction)
    within_ttests = {}
    n_comparisons = (n_superordinates * (n_superordinates - 1)) // 2
    for i, sup1 in enumerate(unique_superordinates):
        for sup2 in unique_superordinates[i+1:]:
            t_stat, p_val = scipy.stats.ttest_ind(
                within_dissimilarities_distributions[sup1],
                within_dissimilarities_distributions[sup2]
            )
            within_ttests[(sup1, sup2)] = {
                't_statistic': t_stat,
                'p_value': p_val * n_comparisons  # Bonferroni correction
            }
    statistical_tests['within_ttests'] = within_ttests
    
    # 3. Compare within vs between dissimilarities
    all_within = np.concatenate(list(within_dissimilarities_distributions.values()))
    all_between = np.concatenate(list(between_dissimilarities_distributions.values()))
    t_stat, p_val = scipy.stats.ttest_ind(all_within, all_between)
    statistical_tests['within_vs_between'] = {
        't_statistic': t_stat,
        'p_value': p_val
    }
    
    return {
        'within_dissimilarities': within_dissimilarities,
        'between_dissimilarities': between_dissimilarities,
        'within_distributions': within_dissimilarities_distributions,
        'between_distributions': between_dissimilarities_distributions,
        'statistical_tests': statistical_tests
    }

def plot_superordinate_dissimilarities(analysis_results, layer_name, dissimilarity_metric, output_dir):
    """
    Create visualizations for superordinate dissimilarity analysis.
    
    Args:
        analysis_results (dict): Results from analyze_superordinate_dissimilarities
        layer_name (str): Name of the layer being analyzed
        dissimilarity_metric (str): dissimilarity metric used
        output_dir (str): Directory to save plots
    """
    # 1. Plot within-superordinate dissimilarities
    plt.figure(figsize=(12, 6))
    within_sims = analysis_results['within_dissimilarities']
    
    # Bar plot
    bars = plt.bar(range(len(within_sims)), 
                  list(within_sims.values()),
                  tick_label=list(within_sims.keys()))
    
    # Add error bars showing standard deviation of distributions
    errors = [np.std(analysis_results['within_distributions'][sup]) 
             for sup in within_sims.keys()]
    plt.errorbar(range(len(within_sims)), list(within_sims.values()),
                yerr=errors, fmt='none', color='black', capsize=5)
    
    plt.title(f'Within-Superordinate dissimilarities - {layer_name}')
    plt.ylabel('Average dissimilarity')
    plt.xticks(rotation=45)
    
    # Add significance annotations
    anova_result = analysis_results['statistical_tests']['within_anova']
    plt.text(0.02, 0.98, 
            f'ANOVA: F={anova_result["f_statistic"]:.2f}, p={anova_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_model_name}_{dissimilarity_metric}_within_dissimilarities_{layer_name}.png'))
    plt.close()
    
    # 2. Plot between-superordinate dissimilarities matrix
    plt.figure(figsize=(10, 8))
    n_superordinates = len(unique_superordinates)
    between_matrix = np.zeros((n_superordinates, n_superordinates))
    
    # Fill the matrix
    for (sup1, sup2), sim in analysis_results['between_dissimilarities'].items():
        i = unique_superordinates.index(sup1)
        j = unique_superordinates.index(sup2)
        between_matrix[i, j] = sim
        between_matrix[j, i] = sim  # Symmetric
    
    # Fill diagonal with within-dissimilarities
    for i, sup in enumerate(unique_superordinates):
        between_matrix[i, i] = analysis_results['within_dissimilarities'][sup]
    
    # Create heatmap
    sns.heatmap(between_matrix,
                xticklabels=unique_superordinates,
                yticklabels=unique_superordinates,
                annot=True,
                fmt='.2f',
                cmap='viridis')
    
    plt.title(f'Superordinate dissimilarity Matrix - {layer_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add within vs between test result
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, -0.15,
            f'Within vs Between t-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_model_name}_{dissimilarity_metric}_between_dissimilarities_{layer_name}.png'))
    plt.close()
    
    # 3. Plot distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Combine all within and between distributions
    all_within = np.concatenate(list(analysis_results['within_distributions'].values()))
    all_between = np.concatenate(list(analysis_results['between_distributions'].values()))
    
    # Create violin plots
    positions = [1, 2]
    violins = plt.violinplot([all_within, all_between], positions, points=100,
                            showmeans=True, showextrema=True)
    
    # Customize violin plots
    for pc in violins['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_alpha(0.7)
    
    plt.title(f'Distribution of Within vs Between dissimilarities - {layer_name}')
    plt.xticks(positions, ['Within\nSuperordinate', 'Between\nSuperordinates'])
    plt.ylabel('dissimilarity')
    
    # Add test statistics
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, 0.98,
            f't-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_model_name}_{dissimilarity_metric}_dissimilarity_distributions_{layer_name}.png'))
    plt.close()

def analyze_animacy_dissimilarities(dissimilarity_matrix, superordinates, labels):
    """
    Analyze dissimilarities within and between animate and inanimate categories.
    
    Args:
        dissimilarity_matrix (np.ndarray): The dissimilarity matrix
        superordinates (list): List of superordinate categories for each sample
        labels (list): List of class labels for each sample
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Define animacy groupings
    animacy_superordinates = {
        "animal": ["ave", "felidae", "fish", "canidae"],
        "nonimal": ["cloth", "land_trans", "kitchen"]
    }
    
    # Create animacy labels for each sample
    animacy_labels = []
    for sup in superordinates:
        if sup in animacy_superordinates["animal"]:
            animacy_labels.append("animal")
        else:
            animacy_labels.append("nonimal")
    
    # Create masks for each animacy category
    animacy_masks = {
        category: np.array([label == category for label in animacy_labels])
        for category in ["animal", "nonimal"]
    }
    
    # Calculate within-animacy dissimilarities
    within_dissimilarities = {}
    within_dissimilarities_distributions = {}
    for category in ["animal", "nonimal"]:
        mask = animacy_masks[category]
        within_sim = dissimilarity_matrix[np.ix_(mask, mask)]
        # Extract upper triangle (excluding diagonal)
        upper_tri = within_sim[np.triu_indices_from(within_sim, k=1)]
        within_dissimilarities[category] = np.mean(upper_tri)
        within_dissimilarities_distributions[category] = upper_tri
    
    # Calculate between-animacy dissimilarities
    mask_animal = animacy_masks["animal"]
    mask_nonimal = animacy_masks["nonimal"]
    between_sim = dissimilarity_matrix[np.ix_(mask_animal, mask_nonimal)]
    between_dissimilarities = np.mean(between_sim)
    between_dissimilarities_distribution = between_sim.flatten()
    
    # Perform statistical tests
    statistical_tests = {}
    
    # 1. Compare animal vs non-animal within-category dissimilarities
    t_stat, p_val = scipy.stats.ttest_ind(
        within_dissimilarities_distributions["animal"],
        within_dissimilarities_distributions["nonimal"]
    )
    statistical_tests['within_comparison'] = {
        't_statistic': t_stat,
        'p_value': p_val
    }
    
    # 2. Compare within vs between dissimilarities
    all_within = np.concatenate([
        within_dissimilarities_distributions["animal"],
        within_dissimilarities_distributions["nonimal"]
    ])
    t_stat, p_val = scipy.stats.ttest_ind(all_within, between_dissimilarities_distribution)
    statistical_tests['within_vs_between'] = {
        't_statistic': t_stat,
        'p_value': p_val
    }
    
    return {
        'within_dissimilarities': within_dissimilarities,
        'between_dissimilarities': between_dissimilarities,
        'within_distributions': within_dissimilarities_distributions,
        'between_distribution': between_dissimilarities_distribution,
        'statistical_tests': statistical_tests
    }

def plot_animacy_dissimilarities(analysis_results, layer_name, dissimilarity_metric, output_dir):
    """
    Create visualizations for animacy-based dissimilarity analysis.
    
    Args:
        analysis_results (dict): Results from analyze_animacy_dissimilarities
        layer_name (str): Name of the layer being analyzed
        dissimilarity_metric (str): dissimilarity metric used
        output_dir (str): Directory to save plots
    """
    # 1. Plot within-animacy dissimilarities
    plt.figure(figsize=(10, 6))
    within_sims = analysis_results['within_dissimilarities']
    
    # Bar plot
    bars = plt.bar(range(len(within_sims)), 
                  list(within_sims.values()),
                  tick_label=list(within_sims.keys()))
    
    # Add error bars showing standard deviation of distributions
    errors = [np.std(analysis_results['within_distributions'][category]) 
             for category in within_sims.keys()]
    plt.errorbar(range(len(within_sims)), list(within_sims.values()),
                yerr=errors, fmt='none', color='black', capsize=5)
    
    plt.title(f'Within-Animacy Category dissimilarities - {layer_name}')
    plt.ylabel('Average dissimilarity')
    
    # Add significance annotations
    test_result = analysis_results['statistical_tests']['within_comparison']
    plt.text(0.02, 0.98, 
            f't-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 
                f'{base_model_name}_{dissimilarity_metric}_within_animacy_dissimilarities_{layer_name}.png'))
    plt.close()
    
    # 2. Plot distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Create violin plots for within and between distributions
    all_within_animal = analysis_results['within_distributions']['animal']
    all_within_nonimal = analysis_results['within_distributions']['nonimal']
    all_between = analysis_results['between_distribution']
    
    positions = [1, 2, 3]
    violins = plt.violinplot([all_within_animal, all_within_nonimal, all_between], 
                           positions, points=100, showmeans=True, showextrema=True)
    
    # Customize violin plots
    colors = ['#D43F3A', '#5CB85C', '#5BC0DE']
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.title(f'Distribution of Within and Between Animacy dissimilarities - {layer_name}')
    plt.xticks(positions, ['Within\nAnimal', 'Within\nNon-animal', 'Between\nCategories'])
    plt.ylabel('dissimilarity')
    
    # Add test statistics
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, 0.98,
            f'Within vs Between t-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 
                f'{base_model_name}_{dissimilarity_metric}_animacy_dissimilarity_distributions_{layer_name}.png'))
    plt.close()

def plot_all_visualizations(all_wnids, layer_results, output_dir):
    """
    Main plotting function that calls individual plotting functions for each visualization.
    
    Args:
        layer_results (dict): Dictionary with layer names as keys and tuples of 
                          (activations_2d, labels, superordinates, dissimilarity_matrix) as values
        output_dir (str): Directory to save the output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, (activations_2d, labels, superordinates, dissimilarity_matrix) in layer_results.items():
        # 1. Plot 2D activations
        plot_2d_activations(activations_2d, superordinates, layer_name, output_dir)
        
        # 2. Plot dissimilarity matrix
        plot_dissimilarity_matrix(dissimilarity_matrix, labels, superordinates, 
                             layer_name, base_model_name, dissimilarity_metric, output_dir)
        
        # # 3. Plot activation distribution
        # raw_activations = get_raw_activations(all_wnids, layer_name)
        # plot_activation_distribution(raw_activations, layer_name, output_dir)
        
        # 4. Plot superordinate dissimilarity analysis
        analysis_results = analyze_superordinate_dissimilarities(dissimilarity_matrix, superordinates, labels)
        plot_superordinate_dissimilarities(analysis_results, layer_name, dissimilarity_metric, output_dir)

        # 5. Plot animacy dissimilarity analysis
        analysis_results = analyze_animacy_dissimilarities(dissimilarity_matrix, superordinates, labels)
        plot_animacy_dissimilarities(analysis_results, layer_name, dissimilarity_metric, output_dir)


def main():
    if not "vit" in base_model_name:
        # Initialize the CNN analyzer
        analyzer = CNNAnalyzer(model_name=base_model_name, dissimilarity_metric=dissimilarity_metric)
    else:
        # Initialize the ViT analyzer
        analyzer = ViTAnalyzer(model_name=base_model_name, dissimilarity_metric=dissimilarity_metric)
    analyzer.get_model_info()
    
    wnid_to_description = load_class_info()
    superordinate_dict, all_wnids = load_superordinate_info()

    layer_results = {}
    for layer_name in layers_to_analyze:
        print(f"Analyzing layer: {layer_name}")
        layer_results[layer_name] = analyzer.analyze_layer(
            layer_name, 
            image_dir,
            actv_output_dir,
            wnid_to_description, 
            superordinate_dict, 
            all_wnids
        )
    
    # Plot all visualizations
    plot_all_visualizations(all_wnids, layer_results, 'figs')

if __name__ == "__main__":
    # Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    base_model_name = 'vit-base-patch16-224'  # vit-base-patch16-224 | dino-vitb16 | vgg16
    actv_output_dir = f"{base_model_name}_actv"
    dissimilarity_metric = 'euclidean'

    num_classes_per_superordinate = 5
    image_dir = '/fast-data20/datasets/ILSVRC/2012/clsloc/val_white'
    unique_superordinates = ["cloth", "land_trans", "ave", "felidae", "fish", "kitchen", "canidae"]
    layers_to_analyze = ["6", "9", "12"] # ViT 13 hidden outputs final layer untrained.
    # layers_to_analyze = ["block4_pool", "block5_pool", "fc2"]
    main()
