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
        res = 1.0 - cosine_similarity(activations)
        # cast to python float64 as json complains about numpy float32
        # float64 is also consistent with `pdist` and `squareform`
        return res.astype('float64')
    else:
        distances = pdist(activations, metric=metric)
        return squareform(distances)
