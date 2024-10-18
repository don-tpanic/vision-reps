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


def plot_all_visualizations(layer_results, output_dir):
    """
    Plot all visualizations for each layer.
    
    Args:
        layer_results (dict): Dictionary with layer names as keys and tuples of 
                              (activations_2d, labels, superordinates, similarity_matrix) as values
        output_dir (str): Directory to save the output figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, (activations_2d, labels, superordinates, similarity_matrix) in layer_results.items():
        # Plot activations (unchanged)
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
        
        # Plot similarity matrix with superordinate and class annotations
        fig, ax = plt.subplots(figsize=(30, 25))
        
        # Sort the similarity matrix and labels based on superordinates and then by class labels
        sorted_indices = np.lexsort((labels, superordinates))
        sorted_similarity_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_superordinates = [superordinates[i] for i in sorted_indices]

        # Create heatmap
        im = ax.imshow(sorted_similarity_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Similarity')

        # Prepare for superordinate and class label annotations
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
                    # Add the last class of the previous superordinate
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

        # Annotate superordinates (left and bottom)
        ax.set_xticks(superordinate_positions)
        ax.set_yticks(superordinate_positions)
        ax.set_xticklabels(unique_superordinates, rotation=90, fontsize=12, ha='center')
        ax.set_yticklabels(unique_superordinates, fontsize=12, va='center')

        # Add lines to group superordinates
        for i in range(1, len(unique_superordinates)):
            pos = (superordinate_positions[i] + superordinate_positions[i-1]) / 2
            ax.axhline(y=pos, color='white', linestyle='--', linewidth=1)
            ax.axvline(x=pos, color='white', linestyle='--', linewidth=1)

        # Annotate class labels (right and top)
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

        # Add lines to group classes
        for pos, _ in class_positions[1:]:
            ax.axhline(y=pos, color='gray', linestyle=':', linewidth=0.5)
            ax.axvline(x=pos, color='gray', linestyle=':', linewidth=0.5)

        plt.title(f'Similarity Matrix ({similarity_metric}) - {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 
            f'{base_model_name}_{similarity_metric}_similarity_matrix_{layer_name}.png'), 
            dpi=300)
        plt.close()

        # Debugging: Print the number of classes per superordinate
        superordinate_class_counts = {}
        for s, l in zip(sorted_superordinates, sorted_labels):
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
    plot_all_visualizations(layer_results, 'figs')


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