import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity


def get_layer_output(model, layer_name):
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def process_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def get_activations(model, img_path):
    img = process_image(img_path)
    return model.predict(img)

def load_class_info():
    df = pd.read_csv('ranked_Imagenet.csv')
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
    for superordinate in superordinates:
        df_super = pd.read_csv(f'{superordinate}_Imagenet.csv')
        wnids = set(df_super['wnid'].values[:num_classes_per_superordinate])
        superordinate_dict[superordinate] = wnids
        all_wnids.update(wnids)
    return superordinate_dict, all_wnids

def compute_pairwise_similarity(activations, metric='cosine'):
    """
    Compute pairwise similarity matrix for the given activations.
    
    Args:
        activations (np.ndarray): 2D array of activations (n_samples, n_features)
        metric (str): Similarity metric to use. Default is 'cosine'.
    
    Returns:
        np.ndarray: 2D array of pairwise similarities
    """
    if metric == 'cosine':
        return cosine_similarity(activations)
    else:
        distances = pdist(activations, metric=metric)
        return 1 - squareform(distances)

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
        activation_dir = os.path.join(actv_output_dir, layer_name, wnid)
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
    
def analyze_layer(layer_name, image_dir, actv_output_dir, wnid_to_description, superordinate_dict, all_wnids, num_images_per_class=50):
    """
    Args:
        layer_name (str): name of the layer to analyze
        image_dir (str): directory containing the images
        actv_output_dir (str): directory to save the activations
        wnid_to_description (dict): dictionary mapping WNID to description
        superordinate_dict (dict): dictionary mapping superordinate categories to WNIDs
        all_wnids (set): set of all WNIDs in the superordinate categories
        num_images_per_class (int): number of images to analyze per class
    
    Returns:
        activations_2d (np.ndarray): 2D array of activations
        labels (list): list of labels (class descriptions) corresponding to the activations
        superordinates (list): list of superordinate categories corresponding to the activations
    
    Notes:
        1. All outputs have the same length and are aligned by index.
        2. Has a check to skip images whose activations have already been saved.
    """
    layer_model = get_layer_output(base_model, layer_name)
    
    activations = []
    labels = []
    superordinates = []
    
    for wnid in all_wnids:
        class_path = os.path.join(image_dir, wnid)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory not found for WNID {wnid}")
            continue
        
        image_files = os.listdir(class_path)[:num_images_per_class]
        
        for img_file in image_files:
            activation_dir = os.path.join(actv_output_dir, layer_name, wnid)
            activation_path = os.path.join(activation_dir, f"{os.path.splitext(img_file)[0]}.npy")
            
            if os.path.exists(activation_path):
                print(f"Loading existing activation for {wnid} | {img_file}")
                flattened_activation = np.load(activation_path)
            else:
                print(f"Processing {wnid} | {img_file}")
                img_path = os.path.join(class_path, img_file)
                activation = get_activations(layer_model, img_path)
                flattened_activation = activation.flatten()
                
                # Create the directory if it doesn't exist
                os.makedirs(activation_dir, exist_ok=True)
                np.save(activation_path, flattened_activation)
            
            activations.append(flattened_activation)
            labels.append(wnid_to_description.get(wnid, wnid))
            
            for superordinate, wnids in superordinate_dict.items():
                if wnid in wnids:
                    superordinates.append(superordinate)
                    break

    activations = np.array(activations)
    
    # Perform PCA
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)

    # Compute pairwise similarity
    similarity_matrix = compute_pairwise_similarity(activations, similarity_metric)
    
    return activations_2d, labels, superordinates, similarity_matrix

def plot_2d_activations(activations_2d, superordinates, layer_name, output_dir):
    """
    Plot 2D activations colored by superordinate category.
    """
    plt.figure(figsize=(20, 20))
    unique_superordinates = list(set(superordinates))
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
    
    plt.title(f'VGG16 Activations - {layer_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'vgg16_activations_{layer_name}.png'), dpi=150)
    plt.close()

def plot_similarity_matrix(similarity_matrix, labels, superordinates, layer_name, base_model_name, similarity_metric, output_dir):
    """
    Plot similarity matrix with superordinate and class annotations.
    """
    fig, ax = plt.subplots(figsize=(30, 25))
    
    # Sort the similarity matrix and labels based on superordinates and then by class labels
    sorted_indices = np.lexsort((labels, superordinates))
    sorted_similarity_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_superordinates = [superordinates[i] for i in sorted_indices]

    # Create heatmap
    im = ax.imshow(sorted_similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Similarity')

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

    plt.title(f'Similarity Matrix ({similarity_metric}) - {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 
        f'{base_model_name}_{similarity_metric}_similarity_matrix_{layer_name}.png'), 
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
    plt.savefig(os.path.join(output_dir, f'activation_distribution_{layer_name}.png'),
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

def analyze_superordinate_similarities(similarity_matrix, superordinates, labels):
    """
    Analyze similarities within and between superordinates.
    
    Args:
        similarity_matrix (np.ndarray): The similarity matrix
        superordinates (list): List of superordinate categories for each sample
        labels (list): List of class labels for each sample
        
    Returns:
        dict: Dictionary containing analysis results
    """
    unique_superordinates = list(set(superordinates))
    n_superordinates = len(unique_superordinates)
    
    # Create masks for each superordinate
    superordinate_masks = {
        sup: np.array([s == sup for s in superordinates])
        for sup in unique_superordinates
    }
    
    # Calculate within-superordinate similarities
    within_similarities = {}
    within_similarities_distributions = {}
    for sup in unique_superordinates:
        mask = superordinate_masks[sup]
        # Get upper triangle of similarity matrix for this superordinate
        within_sim = similarity_matrix[np.ix_(mask, mask)]
        # Extract upper triangle (excluding diagonal)
        upper_tri = within_sim[np.triu_indices_from(within_sim, k=1)]
        within_similarities[sup] = np.mean(upper_tri)
        within_similarities_distributions[sup] = upper_tri
    
    # Calculate between-superordinate similarities
    between_similarities = {}
    between_similarities_distributions = {}
    for i, sup1 in enumerate(unique_superordinates):
        for j, sup2 in enumerate(unique_superordinates[i+1:], i+1):
            mask1 = superordinate_masks[sup1]
            mask2 = superordinate_masks[sup2]
            between_sim = similarity_matrix[np.ix_(mask1, mask2)]
            between_similarities[(sup1, sup2)] = np.mean(between_sim)
            between_similarities_distributions[(sup1, sup2)] = between_sim.flatten()
    
    # Perform statistical tests
    statistical_tests = {}
    
    # 1. ANOVA test for differences among within-superordinate similarities
    within_groups = list(within_similarities_distributions.values())
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
                within_similarities_distributions[sup1],
                within_similarities_distributions[sup2]
            )
            within_ttests[(sup1, sup2)] = {
                't_statistic': t_stat,
                'p_value': p_val * n_comparisons  # Bonferroni correction
            }
    statistical_tests['within_ttests'] = within_ttests
    
    # 3. Compare within vs between similarities
    all_within = np.concatenate(list(within_similarities_distributions.values()))
    all_between = np.concatenate(list(between_similarities_distributions.values()))
    t_stat, p_val = scipy.stats.ttest_ind(all_within, all_between)
    statistical_tests['within_vs_between'] = {
        't_statistic': t_stat,
        'p_value': p_val
    }
    
    return {
        'within_similarities': within_similarities,
        'between_similarities': between_similarities,
        'within_distributions': within_similarities_distributions,
        'between_distributions': between_similarities_distributions,
        'statistical_tests': statistical_tests
    }

def plot_superordinate_similarities(analysis_results, layer_name, output_dir):
    """
    Create visualizations for superordinate similarity analysis.
    
    Args:
        analysis_results (dict): Results from analyze_superordinate_similarities
        layer_name (str): Name of the layer being analyzed
        output_dir (str): Directory to save plots
    """
    # 1. Plot within-superordinate similarities
    plt.figure(figsize=(12, 6))
    within_sims = analysis_results['within_similarities']
    
    # Bar plot
    bars = plt.bar(range(len(within_sims)), 
                  list(within_sims.values()),
                  tick_label=list(within_sims.keys()))
    
    # Add error bars showing standard deviation of distributions
    errors = [np.std(analysis_results['within_distributions'][sup]) 
             for sup in within_sims.keys()]
    plt.errorbar(range(len(within_sims)), list(within_sims.values()),
                yerr=errors, fmt='none', color='black', capsize=5)
    
    plt.title(f'Within-Superordinate Similarities - {layer_name}')
    plt.ylabel('Average Similarity')
    plt.xticks(rotation=45)
    
    # Add significance annotations
    anova_result = analysis_results['statistical_tests']['within_anova']
    plt.text(0.02, 0.98, 
            f'ANOVA: F={anova_result["f_statistic"]:.2f}, p={anova_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'within_similarities_{layer_name}.png'))
    plt.close()
    
    # 2. Plot between-superordinate similarities matrix
    plt.figure(figsize=(10, 8))
    unique_superordinates = list(set(sup for pair in analysis_results['between_similarities'].keys() 
                                   for sup in pair))
    n_superordinates = len(unique_superordinates)
    between_matrix = np.zeros((n_superordinates, n_superordinates))
    
    # Fill the matrix
    for (sup1, sup2), sim in analysis_results['between_similarities'].items():
        i = unique_superordinates.index(sup1)
        j = unique_superordinates.index(sup2)
        between_matrix[i, j] = sim
        between_matrix[j, i] = sim  # Symmetric
    
    # Fill diagonal with within-similarities
    for i, sup in enumerate(unique_superordinates):
        between_matrix[i, i] = analysis_results['within_similarities'][sup]
    
    # Create heatmap
    sns.heatmap(between_matrix,
                xticklabels=unique_superordinates,
                yticklabels=unique_superordinates,
                annot=True,
                fmt='.2f',
                cmap='viridis')
    
    plt.title(f'Superordinate Similarity Matrix - {layer_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add within vs between test result
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, -0.15,
            f'Within vs Between t-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'between_similarities_{layer_name}.png'))
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
    
    plt.title(f'Distribution of Within vs Between Similarities - {layer_name}')
    plt.xticks(positions, ['Within\nSuperordinate', 'Between\nSuperordinates'])
    plt.ylabel('Similarity')
    
    # Add test statistics
    test_result = analysis_results['statistical_tests']['within_vs_between']
    plt.text(0.02, 0.98,
            f't-test: t={test_result["t_statistic"]:.2f}, p={test_result["p_value"]:.3e}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'similarity_distributions_{layer_name}.png'))
    plt.close()

def plot_all_visualizations(all_wnids, layer_results, output_dir):
    """
    Main plotting function that calls individual plotting functions for each visualization.
    
    Args:
        layer_results (dict): Dictionary with layer names as keys and tuples of 
                          (activations_2d, labels, superordinates, similarity_matrix) as values
        output_dir (str): Directory to save the output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, (activations_2d, labels, superordinates, similarity_matrix) in layer_results.items():
        # 1. Plot 2D activations
        plot_2d_activations(activations_2d, superordinates, layer_name, output_dir)
        
        # 2. Plot similarity matrix
        plot_similarity_matrix(similarity_matrix, labels, superordinates, 
                             layer_name, base_model_name, similarity_metric, output_dir)
        
        # 3. Plot activation distribution
        raw_activations = get_raw_activations(all_wnids, layer_name)
        plot_activation_distribution(raw_activations, layer_name, output_dir)
        
        # Print debugging information
        print_superordinate_class_counts(superordinates, labels)

        # 4. Plot superordinate similarity analysis
        analysis_results = analyze_superordinate_similarities(similarity_matrix, superordinates, labels)
        plot_superordinate_similarities(analysis_results, layer_name, output_dir)

def print_superordinate_class_counts(superordinates, labels):
    """
    Print debugging information about the number of classes per superordinate.
    """
    superordinate_class_counts = {}
    for s, l in zip(superordinates, labels):
        if s not in superordinate_class_counts:
            superordinate_class_counts[s] = set()
        superordinate_class_counts[s].add(l)
    
    for s, classes in superordinate_class_counts.items():
        print(f"Superordinate {s} has {len(classes)} classes: {', '.join(classes)}")


def main():
    wnid_to_description = load_class_info()
    superordinate_dict, all_wnids = load_superordinate_info()

    layer_results = {}
    for layer_name in layers_to_analyze:
        print(f"Analyzing layer: {layer_name}")
        layer_results[layer_name] = analyze_layer(
            layer_name, 
            image_dir,
            actv_output_dir,
            wnid_to_description, 
            superordinate_dict, 
            all_wnids
        )
    
    # Plot all visualizations at once
    plot_all_visualizations(all_wnids, layer_results, 'figs')


if __name__ == "__main__":
    base_model_name = 'vgg16'
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=True)
    base_model.summary()
    actv_output_dir = f"{base_model_name}_actv"
    similarity_metric = 'cosine'

    num_classes_per_superordinate = 5
    image_dir = '/fast-data20/datasets/ILSVRC/2012/clsloc/val_white'
    superordinates = ["cloth", "land_trans", "ave", "felidae", "fish", "kitchen", "canidae"]
    layers_to_analyze = ["block4_pool", "block5_pool", "fc2"]

    main()