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

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=True)
base_model.summary()

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

def load_superordinate_info(superordinates):
    """
    Args:   
        superordinates (list): list of superordinate categories (str) to analyze
    
    Returns:
        superordinate_dict (dict): dictionary of superordinate categories to WNIDs
        all_wnids (set): set of all WNIDs in the superordinate categories
    """
    superordinate_dict = {}
    all_wnids = set()
    for superordinate in superordinates:
        df_super = pd.read_csv(f'{superordinate}_Imagenet.csv')
        wnids = set(df_super['wnid'].values[:2])  # Keep to 2 classes per superordinate.
        superordinate_dict[superordinate] = wnids
        all_wnids.update(wnids)
    return superordinate_dict, all_wnids

def analyze_layer(layer_name, image_dir, wnid_to_description, superordinate_dict, all_wnids, num_images_per_class=50):
    """
    Args:
        layer_name (str): name of the layer to analyze
        image_dir (str): directory containing the images
        wnid_to_description (dict): dictionary mapping WNID to description
        superordinate_dict (dict): dictionary mapping superordinate categories to WNIDs
        all_wnids (set): set of all WNIDs in the superordinate categories
        num_images_per_class (int): number of images to analyze per class
    
    Returns:
        activations_2d (np.ndarray): 2D array of activations
        labels (list): list of labels (class descriptions) corresponding to the activations
        superordinates (list): list of superordinate categories corresponding to the activations
    
    All outputs have the same length and are aligned by index.
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
            print(f"Processing {wnid} | {img_file}")
            img_path = os.path.join(class_path, img_file)
            activation = get_activations(layer_model, img_path)
            activations.append(activation.flatten())
            labels.append(wnid_to_description.get(wnid, wnid))
            
            for superordinate, wnids in superordinate_dict.items():
                if wnid in wnids:
                    superordinates.append(superordinate)
                    break

    activations = np.array(activations)
    
    # Perform PCA
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)
    
    return activations_2d, labels, superordinates

def plot_activations(activations_2d, labels, superordinates, layer_name):
    plt.figure(figsize=(20, 20))
    
    unique_superordinates = list(set(superordinates))
    color_map = matplotlib.colormaps["tab20"]
    colors = {s: color_map(i/len(unique_superordinates)) for i, s in enumerate(unique_superordinates)}
    
    for superordinate in unique_superordinates:
        mask = np.array(superordinates) == superordinate  # superordinate-level mask
        plt.scatter(activations_2d[mask, 0], activations_2d[mask, 1], 
                    c=[colors[superordinate]], label=superordinate, alpha=0.7)
    
    plt.title(f'VGG16 Activations - {layer_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'vgg16_activations_{layer_name}.png', dpi=300)
    plt.close()

def main():
    image_dir = '/fast-data20/datasets/ILSVRC/2012/clsloc/val_white'
    superordinates = ["cloth", "land_trans", "ave", "felidae", "fish", "kitchen", "canidae"]

    wnid_to_description = load_class_info()
    superordinate_dict, all_wnids = load_superordinate_info(superordinates)
    
    layers_to_analyze = ["block4_pool", "block5_pool", "fc2"]
    
    for layer_name in layers_to_analyze:
        print(f"Analyzing layer: {layer_name}")
        activations_2d, labels, superordinates = analyze_layer(layer_name, image_dir, wnid_to_description, superordinate_dict, all_wnids)
        plot_activations(activations_2d, labels, superordinates, layer_name)

if __name__ == "__main__":
    main()