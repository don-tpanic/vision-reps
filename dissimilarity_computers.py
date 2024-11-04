from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

def compute_pairwise_dissimilarity(activations, metric='cosine'):
    """
    Compute pairwise dissimilarity matrix for the given activations.
    
    Args:
        activations (np.ndarray): 2D array of activations (n_samples, n_features)
        metric (str): Dissimilarity metric to use. Default is 'cosine' distance.
    
    Returns:
        np.ndarray: 2D array of pairwise dissimilarities
    """
    if metric == 'cosine':
        return 1.0 - cosine_similarity(activations)
    else:
        distances = pdist(activations, metric=metric)
        return squareform(distances)
