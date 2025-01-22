import numpy as np
from tqdm import tqdm


def featurenormalize(feature_vector):
    """
        Method to get a normalized feature vector i.e. values between 0 and 1. 
        Args:
            feature_vector(np.ndarray): n_samples x n_features vectors.
        Return:
            np.ndarray of shape n_samples x n_features vector consisting of normalized values.
    """
    if np.isnan(feature_vector).any():
        feature_vector = np.nan_to_num(feature_vector)
    norm = np.linalg.norm(feature_vector)
    return feature_vector / norm if norm != 0 else feature_vector

def euclidean_distance_calculator(x, y):
    return np.linalg.norm(x - y)


def calculate_stress(X_original, X_latent_dim):
    """
        Funtion to find out the average stress between data in original space and feature reduced space.
        Args:
            X_original(np.array): dataset represent in original space.
        Return:
            X_latent_dim(np.array): dataset represented in feature reduced latent space.
    """
    X_original = np.array(X_original)
    X_latent_dim = np.array(X_latent_dim)

    n_samples = X_original.shape[0]
    stress_sum = 0.0
    distance_sum = 0.0

    for i in tqdm(range(n_samples)):
        for j in range(i + 1, n_samples):
            d_ij = np.linalg.norm(X_original[i] - X_original[j])

            delta_ij = np.linalg.norm(X_latent_dim[i] - X_latent_dim[j])

            stress_sum += (d_ij - delta_ij)**2
            distance_sum += d_ij**2

    stress = np.sqrt(stress_sum / distance_sum)

    return stress

def pca_mle(X):
    """
    Perform PCA for dimensionality reduction using maximum likelihood estimation.

    Parameters:
     X(np.array): The original dataset.

    Returns:
    - X_reduced(np.array): Feature Reduced dataset.
    - components: numpy array, shape (n_features, n_components)
      The principal components.
    """

    X = np.array(X)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    threshold = 0.99
    num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    components = eigenvectors[:, :num_components]

    X_reduced = np.dot(X_centered, components)

    return X_reduced

