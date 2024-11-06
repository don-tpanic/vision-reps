import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy

import stats
from utils.plotting import *
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
        within_dissimilarities_distributions[sup] = list(upper_tri)
    
    # Calculate between-superordinate dissimilarities
    between_dissimilarities = {}
    between_dissimilarities_distributions = {}
    for i, sup1 in enumerate(unique_superordinates):
        for j, sup2 in enumerate(unique_superordinates[i+1:], i+1):
            mask1 = superordinate_masks[sup1]
            mask2 = superordinate_masks[sup2]
            between_sim = dissimilarity_matrix[np.ix_(mask1, mask2)]
            between_dissimilarities[f"{sup1},{sup2}"] = np.mean(between_sim)
            between_dissimilarities_distributions[f"{sup1},{sup2}"] = list(between_sim.flatten())
    
    # Perform statistical tests
    statistical_tests = {}
    
    # 1. ANOVA test for differences among within-superordinate dissimilarities
    within_groups = list(within_dissimilarities_distributions.values())
    f_stat, p_value = scipy.stats.f_oneway(*within_groups)
    statistical_tests['within_anova'] = {
        'f_statistic': f_stat,
        'p_value': p_value
    }
        
    # 2. Compare within vs between dissimilarities
    # Record,
    # 1) mean difference
    # 2) bootstrapped confidence intervals
    # 3) t-test results (t-statistic and p-value)
    all_within = np.concatenate(list(within_dissimilarities_distributions.values()))
    all_between = np.concatenate(list(between_dissimilarities_distributions.values()))
    stats_results = stats.compare_distributions_mean_diff_and_bootstrap_ci(
        all_within, all_between, n_bootstrap=100, ci_level=0.95, random_state=42)
    t_stat, p_value = scipy.stats.ttest_ind(all_within, all_between)

    statistical_tests['within_vs_between'] = {
        'mean_diff': stats_results['mean_diff'],
        'ci_lower': stats_results['ci_lower'],
        'ci_upper': stats_results['ci_upper'],
        't_statistic': t_stat,
        'p_value': p_value
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
    for str_sup1_sup2, sim in analysis_results['between_dissimilarities'].items():
        sup1, sup2 = str_sup1_sup2.split(',')
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
    colors = ['#D43F3A', '#5CB85C']
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.title(f'Distribution of Within vs Between dissimilarities - {layer_name}')
    plt.xticks(positions, ['Within\nSuperordinate', 'Between\nSuperordinates'])
    plt.ylabel('dissimilarity')
    
    # Add test statistics
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, 0.98,
            f'Mean Difference: {test_result["mean_diff"]:.3f}\nCI: [{test_result["ci_lower"]:.3f}, {test_result["ci_upper"]:.3f}]\n' \
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
        within_dissimilarities_distributions[category] = list(upper_tri)
    
    # Calculate between-animacy dissimilarities
    mask_animal = animacy_masks["animal"]
    mask_nonimal = animacy_masks["nonimal"]
    between_sim = dissimilarity_matrix[np.ix_(mask_animal, mask_nonimal)]
    between_dissimilarities = np.mean(between_sim)
    between_dissimilarities_distribution = list(between_sim.flatten())
    
    # Perform statistical tests
    statistical_tests = {}
    
    # 1. Compare within vs between dissimilarities
    all_within = np.concatenate([
        within_dissimilarities_distributions["animal"],
        within_dissimilarities_distributions["nonimal"]
    ])
    stats_results = stats.compare_distributions_mean_diff_and_bootstrap_ci(
        all_within, between_dissimilarities_distribution, n_bootstrap=100, ci_level=0.95, random_state=42)
    t_stat, p_val = scipy.stats.ttest_ind(all_within, between_dissimilarities_distribution)
    statistical_tests['within_vs_between'] = {
        'mean_diff': stats_results['mean_diff'],
        'ci_lower': stats_results['ci_lower'],
        'ci_upper': stats_results['ci_upper'],
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
    # 1. Plot distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Create violin plots for within and between distributions
    all_within_animal = analysis_results['within_distributions']['animal']
    all_between = analysis_results['between_distribution']
    
    positions = [1, 2]
    violins = plt.violinplot([all_within_animal, all_between], 
                           positions, points=100, showmeans=True, showextrema=True)
    
    # Customize violin plots
    colors = ['#D43F3A', '#5CB85C']
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.title(f'Distribution of Within and Between Animacy dissimilarities - {layer_name}')
    plt.xticks(positions, ['Within\nAnimal', 'Between\nCategories'])
    plt.ylabel('dissimilarity')
    
    # Add test statistics
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, 0.98,
            f'Mean Difference: {test_result["mean_diff"]:.4f}\nCI: [{test_result["ci_lower"]:.4f}, {test_result["ci_upper"]:.4f}]\n' \
            f'Within vs Between t-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 
                f'{base_model_name}_{dissimilarity_metric}_animacy_dissimilarity_distributions_{layer_name}.png'))
    plt.close()

def plot_animacy_mean_diff_over_layers(layers_to_analyze, dissimilarity_metric, results_dir):
    """
    Plot how animacy mean_diff between within and between animal-nonimal
    changes over layers. fill_between with ci.

    Args:
        layers_to_analyze (list): List of layer names
        dissimilarity_metric (str): Dissimilarity metric used
        results_dir (str): Directory containing analysis results
    """
    layer_mean_diff_and_ci = []
    for layer_name in layers_to_analyze:
        layer_results = load_layer_results(layer_name, dissimilarity_metric, results_dir)
        animacy_results = layer_results['animacy_analysis']
        test_result = animacy_results['statistical_tests']['within_vs_between']
        layer_mean_diff_and_ci.append((layer_name, test_result['mean_diff'], test_result['ci_lower'], test_result['ci_upper']))

    layer_names, mean_diffs, ci_lowers, ci_uppers = zip(*layer_mean_diff_and_ci)
    plt.figure(figsize=(12, 6))
    plt.plot(layer_names, mean_diffs, marker='o', label='Mean Difference')
    plt.fill_between(layer_names, ci_lowers, ci_uppers, alpha=0.3, label='95% CI')
    plt.xlabel('Layer')
    plt.ylabel(f'Mean Difference ({dissimilarity_metric})')
    plt.title('Animacy Mean Difference between Within and Between Dissimilarities')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figs', f'{base_model_name}_{dissimilarity_metric}_animacy_mean_diff_over_layers.png'))
    plt.close()


def save_layer_results(results, dissimilarity_metric, layer_name, output_dir):
    """
    Save all analysis results for a layer.
    
    Args:
        results (dict): Results from analyze_layer_results
        dissimilarity_metric (str): Dissimilarity metric used
        layer_name (str): Name of the layer
        output_dir (str): Directory to save results
    """
    layer_dir = os.path.join(output_dir, dissimilarity_metric, layer_name)
    os.makedirs(layer_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(layer_dir, 'activations_2d.npy'), results['activations_2d'])
    np.save(os.path.join(layer_dir, 'dissimilarity_matrix.npy'), results['dissimilarity_matrix'])
    
    # Save lists and analysis results as JSON
    with open(os.path.join(layer_dir, 'metadata.json'), 'w') as f:
        json_results = {
            'labels': results['labels'],
            'superordinates': results['superordinates'],
            'superordinate_analysis': results['superordinate_analysis'],
            'animacy_analysis': results['animacy_analysis']
        }
        json.dump(json_results, f, indent=2)

def load_layer_results(layer_name, dissimilarity_metric, results_dir):
    """
    Load all analysis results for a layer.
    
    Args:
        layer_name (str): Name of the layer
        dissimilarity_metric (str): Dissimilarity metric used
        results_dir (str): Directory containing results
        
    Returns:
        dict: Complete analysis results
    """
    layer_dir = os.path.join(results_dir, dissimilarity_metric, layer_name)
    
    # Load numpy arrays
    activations_2d = np.load(os.path.join(layer_dir, 'activations_2d.npy'))
    dissimilarity_matrix = np.load(os.path.join(layer_dir, 'dissimilarity_matrix.npy'))
    
    # Load JSON data
    with open(os.path.join(layer_dir, 'metadata.json'), 'r') as f:
        json_results = json.load(f)
    
    return {
        'activations_2d': activations_2d,
        'dissimilarity_matrix': dissimilarity_matrix,
        'labels': json_results['labels'],
        'superordinates': json_results['superordinates'],
        'superordinate_analysis': json_results['superordinate_analysis'],
        'animacy_analysis': json_results['animacy_analysis']
    }

def analyze_layer_results(activations_2d, labels, superordinates, dissimilarity_matrix):
    """
    Analyze layer results and compute all metrics at once.
    
    Args:
        activations_2d (np.ndarray): 2D projection of activations
        labels (list): Class labels
        superordinates (list): Superordinate categories
        dissimilarity_matrix (np.ndarray): Matrix of pairwise dissimilarities
        
    Returns:
        dict: Complete analysis results
    """
    # Analyze superordinate dissimilarities
    superordinate_results = analyze_superordinate_dissimilarities(
        dissimilarity_matrix, superordinates, labels)
    
    # Analyze animacy dissimilarities
    animacy_results = analyze_animacy_dissimilarities(
        dissimilarity_matrix, superordinates, labels)
    
    return {
        'activations_2d': activations_2d,
        'labels': labels,
        'superordinates': superordinates,
        'dissimilarity_matrix': dissimilarity_matrix,
        'superordinate_analysis': superordinate_results,
        'animacy_analysis': animacy_results
    }

def plot_all_visualizations(all_wnids, results_dir, output_dir):
    """
    Plot all visualizations using saved results.
    
    Args:
        all_wnids (set): Set of all WNIDs
        results_dir (str): Directory containing analysis results
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name in layers_to_analyze:
        print(f"Plotting layer: {layer_name}")
        # Load results
        layer_results = load_layer_results(layer_name, dissimilarity_metric, results_dir)
        
        # Plot 2D activations
        plot_2d_activations(
            layer_results['activations_2d'],
            layer_results['superordinates'],
            layer_name,
            output_dir
        )
        
        # Plot dissimilarity matrix
        plot_dissimilarity_matrix(
            layer_results['dissimilarity_matrix'],
            layer_results['labels'],
            layer_results['superordinates'],
            layer_name,
            base_model_name,
            dissimilarity_metric,
            output_dir
        )
        
        # Plot superordinate dissimilarity analysis
        plot_superordinate_dissimilarities(
            layer_results['superordinate_analysis'],
            layer_name,
            dissimilarity_metric,
            output_dir
        )
        
        # Plot animacy dissimilarity analysis
        plot_animacy_dissimilarities(
            layer_results['animacy_analysis'],
            layer_name,
            dissimilarity_metric,
            output_dir
        )

    # Plot how animacy mean_diff between within and between animal-nonimal
    # changes over layers. fill_between with ci.
    plot_animacy_mean_diff_over_layers(layers_to_analyze, dissimilarity_metric, results_dir)


def main():
    if not "vit" in base_model_name:
        analyzer = CNNAnalyzer(model_name=base_model_name, dissimilarity_metric=dissimilarity_metric)
    else:
        analyzer = ViTAnalyzer(model_name=base_model_name, dissimilarity_metric=dissimilarity_metric)
    analyzer.get_model_info()
    
    wnid_to_description = load_class_info()
    superordinate_dict, all_wnids = load_superordinate_info()
    
    # Analysis phase
    results_dir = os.path.join('results', base_model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    for layer_name in layers_to_analyze:
        print(f"Analyzing layer: {layer_name}")
        layer_dir = os.path.join(results_dir, dissimilarity_metric, layer_name)
        if not os.path.exists(layer_dir):
            # Get basic layer-wise activations and dissimilarity matrices
            # which we do further analysis on.
            activations_2d, labels, superordinates, dissimilarity_matrix = analyzer.analyze_layer(
                layer_name, 
                image_dir,
                actv_output_dir,
                wnid_to_description, 
                superordinate_dict, 
                all_wnids
            )
        
            # Compute all analysis results
            layer_results = analyze_layer_results(
                activations_2d, labels, superordinates, dissimilarity_matrix)
            
            # Save results
            save_layer_results(layer_results, dissimilarity_metric, layer_name, results_dir)
        else:
            print(f"Results already exist for layer: {layer_name}, skip to plotting.")
    
    # Plotting phase
    plot_all_visualizations(all_wnids, results_dir, "figs")

if __name__ == "__main__":
    # Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    base_model_name = 'vit-base-patch16-224'  # vit-base-patch16-224 | dino-vitb16 | vgg16
    actv_output_dir = f"{base_model_name}_actv"
    dissimilarity_metric = 'euclidean'

    num_classes_per_superordinate = 5
    image_dir = '/fast-data20/datasets/ILSVRC/2012/clsloc/val_white'
    unique_superordinates = ["cloth", "land_trans", "ave", "felidae", "fish", "kitchen", "canidae"]
    layers_to_analyze = ["3", "6", "9", "12"] # ViT 13 hidden outputs final layer untrained.
    # layers_to_analyze = ["block4_pool", "block5_pool", "fc2"]
    main()
