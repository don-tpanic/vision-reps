from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

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
