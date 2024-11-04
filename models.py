import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor

from dissimilarity_computers import compute_pairwise_dissimilarity


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
            tuple: (activations_2d, labels, superordinates, dissimilarity_matrix)
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

        # Compute dissimilarity matrix
        dissimilarity_matrix = compute_pairwise_dissimilarity(activations)
        
        return activations_2d, labels, superordinates, dissimilarity_matrix
    
    def get_model_info(self):
        """Print model summary."""
        self.model.summary()


class ViTAnalyzer:
    """
    A class for analyzing Vision Transformer (ViT) model representations.
    
    Attributes:
        model_name (str): Name of the ViT model ('vit-base', 'vit-large', 'dino-vit-small', 'dino-vit-base')
        model: The loaded model
        preprocessor: The appropriate image preprocessor for the model
        device (torch.device): Device to run the model on
    """
    
    SUPPORTED_MODELS = {
        'vit-base-patch16-224': ('google/vit-base-patch16-224', (224, 224)),  # https://huggingface.co/google/vit-base-patch16-224
        'dino-vitb16': ('facebook/dino-vitb16', (224, 224)),         # https://huggingface.co/facebook/dino-vitb16
    }
    
    def __init__(self, model_name='vit-base-patch16-224', device=None):
        """
        Initialize the ViT analyzer.
        
        Args:
            model_name (str): Name of the model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
            
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_path, self.input_shape = self.SUPPORTED_MODELS[model_name]
        
        # Load model and image preprocesser based on model type
        self.preprocesser = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTModel.from_pretrained(model_path)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def process_image(self, img_path):
        """
        Load and preprocess an image for the model.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load and preprocess image using the feature extractor
        inputs = self.preprocesser(
            images=Image.open(img_path),
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def get_layer_output(self, layer_name, encoded_img):
        """
        Get outputs for a specific transformer layer.
        
        Args:
            layer_name (int): Index of the transformer layer
            encoded_img: Preprocessed image tensor
            
        Returns:
            np.ndarray: Layer outputs
        """
        with torch.no_grad():
            outputs = self.model(**encoded_img, output_hidden_states=True)
            # Get the specified layer's hidden states
            layer_output = outputs.hidden_states[int(layer_name)]
            return layer_output.cpu().numpy()
        
    def analyze_layer(self, layer_name, image_dir, actv_output_dir, wnid_to_description, 
                     superordinate_dict, all_wnids, num_images_per_class=50):
        """
        Analyze a specific layer's representations.
        
        Args:
            layer_name (int): Index of the transformer layer to analyze
            image_dir (str): Directory containing the images
            actv_output_dir (str): Directory to save the activations
            wnid_to_description (dict): Mapping of WNIDs to descriptions
            superordinate_dict (dict): Mapping of superordinate categories to WNIDs
            all_wnids (set): Set of all WNIDs to analyze
            num_images_per_class (int): Number of images to analyze per class
            
        Returns:
            tuple: (activations_2d, labels, superordinates, dissimilarity_matrix)
        """
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
                activation_dir = os.path.join(actv_output_dir, f"layer_{layer_name}", wnid)
                activation_path = os.path.join(activation_dir, f"{os.path.splitext(img_file)[0]}.npy")
                
                if os.path.exists(activation_path):
                    print(f"Loading existing activation for {wnid} | {img_file}")
                    cls_token_features = np.load(activation_path)
                else:
                    print(f"Processing {wnid} | {img_file}")
                    img_path = os.path.join(class_path, img_file)
                    encoded_img = self.process_image(img_path)
                    layer_output = self.get_layer_output(layer_name, encoded_img)
                    # Extract CLS token features
                    cls_token_features = layer_output[0, 0]
                    
                    os.makedirs(activation_dir, exist_ok=True)
                    np.save(activation_path, cls_token_features)
                
                activations.append(cls_token_features)
                labels.append(wnid_to_description.get(wnid, wnid))
                
                for superordinate, wnids in superordinate_dict.items():
                    if wnid in wnids:
                        superordinates.append(superordinate)
                        break
        
        activations = np.array(activations)
        
        # Perform PCA
        pca = PCA(n_components=2)
        activations_2d = pca.fit_transform(activations)
        
        # Compute dissimilarity matrix
        dissimilarity_matrix = compute_pairwise_dissimilarity(activations)
        
        return activations_2d, labels, superordinates, dissimilarity_matrix
    
    def get_model_info(self):
        """Print model information."""
        print(f"Model name: {self.model_name}")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")
        print(f"Hidden size: {self.model.config.hidden_size}")
        print(f"Number of attention heads: {self.model.config.num_attention_heads}")
        print(f"Patch size: {self.model.config.patch_size}")
        import time 
        time.sleep(5)