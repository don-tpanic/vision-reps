import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

from similarity_computers import compute_pairwise_similarity

class CNNAnalyzer:
    """
    A class for analyzing CNN model activations and representations.
    
    Attributes:
        model_name (str): Name of the CNN model ('vgg16', 'resnet50', 'inception_v3')
        model: The loaded Keras model
        preprocess_input: The appropriate preprocessing function for the model
        input_shape (tuple): Expected input shape for the model
    """
    
    SUPPORTED_MODELS = {
        'vgg16': (VGG16, vgg_preprocess, (224, 224)),
        'resnet50': (ResNet50, resnet_preprocess, (224, 224)),
        'inception_v3': (InceptionV3, inception_preprocess, (299, 299))
    }
    
    def __init__(self, model_name='vgg16', weights='imagenet', include_top=True):
        """
        Initialize the CNN analyzer.
        
        Args:
            model_name (str): Name of the model to use
            weights (str): Weight initialization strategy
            include_top (bool): Whether to include the top layers
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
            
        self.model_name = model_name
        model_class, self.preprocess_input, self.input_shape = self.SUPPORTED_MODELS[model_name]
        self.model = model_class(weights=weights, include_top=include_top)
        
    def get_layer_output_model(self, layer_name):
        """Get a model that outputs the specified layer's activations."""
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
    
    def process_image(self, img_path):
        """
        Load and preprocess an image for the model.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        img = load_img(img_path, target_size=self.input_shape)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return self.preprocess_input(img_array)
    
    def get_activations(self, layer_model, img_path):
        """
        Get activations for a specific layer and image.
        
        Args:
            layer_model: Model configured to output specific layer
            img_path (str): Path to the image file
            
        Returns:
            np.ndarray: Layer activations
        """
        img = self.process_image(img_path)
        return layer_model.predict(img)
    
    def analyze_layer(self, layer_name, image_dir, actv_output_dir, wnid_to_description, 
                     superordinate_dict, all_wnids, num_images_per_class=50):
        """
        Analyze a specific layer's representations.
        
        Args:
            layer_name (str): Name of the layer to analyze
            image_dir (str): Directory containing the images
            actv_output_dir (str): Directory to save the activations
            wnid_to_description (dict): Mapping of WNIDs to descriptions
            superordinate_dict (dict): Mapping of superordinate categories to WNIDs
            all_wnids (set): Set of all WNIDs to analyze
            num_images_per_class (int): Number of images to analyze per class
            
        Returns:
            tuple: (activations_2d, labels, superordinates, similarity_matrix)
        """
        layer_model = self.get_layer_output_model(layer_name)
        
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
                    activation = self.get_activations(layer_model, img_path)
                    flattened_activation = activation.flatten()
                    
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

        # Compute similarity matrix
        similarity_matrix = compute_pairwise_similarity(activations)
        
        return activations_2d, labels, superordinates, similarity_matrix
    
    def get_model_summary(self):
        """Print model summary."""
        self.model.summary()