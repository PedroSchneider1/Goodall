# template_matching_sys.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from typing import List
from dtaidistance import dtw

class TemplateMatchingSystem:
    """
    Advanced template matching system for comparing CWT features against known templates.
    
    This system provides multiple similarity metrics and combination strategies to
    robustly identify monkey strikes based on learned templates.
    """

    def __init__(self, normalize_features: bool = True):
        """
        Initialize the system

        Args:
            normalize_features (bool, optional): Wheter to normalize features before comparison,
            usually recommended for CWT features. Defaults to True.
        """
        self.templates = []
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.is_fitted = False
    
    def add_templates(self, template_features: List[np.ndarray]):
        """
        Add your templates CWT feature vectors to the system.
        
        Args:
            template_features (List[np.ndarray]): List of numpy arrays, each containing CWT features for one template
        """
        self.templates = [np.array(template) for template in template_features]

        print(f"Added {len(self.templates)} templates:")

        # If normalization is enabled, fit the scaler on template data
        if self.normalize_features and len(self.templates) > 0:
            template_matrix = np.array(self.templates)
            print(f"Original template_matrix shape: {template_matrix.shape}")
            # Reshape to 2D: (n_templates, n_features)
            if template_matrix.ndim > 2:
                template_matrix = template_matrix.reshape(template_matrix.shape[0], -1) # original shape (n_templates, n_events, n_features)
            print(f"Reshaped template_matrix shape: {template_matrix.shape}")
            self.scaler.fit(template_matrix)
            self.templates = [self.scaler.transform(template.reshape(1, -1)).flatten() 
                            for template in template_matrix]
            self.is_fitted = True
            print("Templates normalized using StandardScaler\n")

    def calculate_cosine_similarities(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between candidates and all templates.
        
        Cosine similarity measures the cosine of the angle between two vectors.
        It's excellent for comparing patterns regardless of their magnitude.
        Values range from -1 to 1, where 1 means identical patterns.
        
        Returns:
            ndarray (n_candidates, n_templates) with similarity scores
        """
        if self.normalize_features and self.is_fitted:
            candidate_matrix = np.array(candidate_features)
            print(f"Original candidate_matrix shape: {candidate_matrix.shape}")
            # Reshape to 2D: (n_events, n_features)
            if candidate_matrix.ndim > 2:
                candidate_matrix = candidate_matrix.reshape(candidate_matrix.shape[1], -1) # original shape (n_candidates, n_events, n_features)
            print(f"Reshaped candidate_matrix shape: {candidate_matrix.shape}")
            candidate_features = self.scaler.transform(candidate_matrix)
        print("Candidate features normalized using StandardScaler\n")
        
        # Calculate cosine similarity between each candidate and each template

        # Matrix MxN where M is number of candidates and N is number of templates
        # Each row corresponds to a candidate, each column to a template
        # Each element (i,j) is the cosine similarity score between candidate i and template j

        similarities = np.zeros((len(candidate_features), len(self.templates)))
        
        for i, template in enumerate(self.templates):
            template_matrix = template.reshape(1, -1)
            candidate_similarities = cosine_similarity(candidate_features, template_matrix).flatten()
            similarities[:, i] = candidate_similarities
        
        return similarities

    def calculate_correlation_similarities(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Calculate Pearson correlation coefficient between candidates and templates.
        
        Pearson correlation measures linear correlation between two vectors.
        Values range from -1 to 1, where 1 means perfect positive correlation.
        
        Returns:
            Array of shape (n_candidates, n_templates) with correlation scores
        """
        if self.normalize_features and self.is_fitted:
            candidate_matrix = np.array(candidate_features)
            print(f"Original candidate_matrix shape: {candidate_matrix.shape}")
            # Reshape to 2D: (n_events, n_features)
            if candidate_matrix.ndim > 2:
                candidate_matrix = candidate_matrix.reshape(candidate_matrix.shape[1], -1) # original shape (n_candidates, n_events, n_features)
            print(f"Reshaped candidate_matrix shape: {candidate_matrix.shape}")
            candidate_features = self.scaler.transform(candidate_matrix)
        print("Candidate features normalized using StandardScaler\n")

        similarities = np.zeros((len(candidate_features), len(self.templates)))

        for i, template in enumerate(self.templates):
            for j, candidate in enumerate(candidate_features):
                corr, _ = pearsonr(candidate, template)
                similarities[j, i] = corr
        return similarities

    def calculate_euclidean_similarities(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance-based similarity between candidates and templates.
        
        Euclidean distance measures straight-line distance in feature space.
        We convert to similarity by using 1/(1+distance) so higher values mean more similar.
        
        Returns:
            Array of shape (n_candidates, n_templates) with similarity scores
        """
        if self.normalize_features and self.is_fitted:
            candidate_matrix = np.array(candidate_features)
            print(f"Original candidate_matrix shape: {candidate_matrix.shape}")
            # Reshape to 2D: (n_events, n_features)
            if candidate_matrix.ndim > 2:
                candidate_matrix = candidate_matrix.reshape(candidate_matrix.shape[1], -1) # original shape (n_candidates, n_events, n_features)
            print(f"Reshaped candidate_matrix shape: {candidate_matrix.shape}")
            candidate_features = self.scaler.transform(candidate_matrix)
        print("Candidate features normalized using StandardScaler\n")
        
        similarities = np.zeros((len(candidate_features), len(self.templates)))
        
        for i, template in enumerate(self.templates):
            template_matrix = template.reshape(1, -1)
            
            # Use Euclidean distance
            distances = euclidean_distances(candidate_features, template_matrix).flatten()
            
            # Convert distance to similarity: smaller distance = higher similarity
            similarities[:, i] = 1 / (1 + distances)
        
        return similarities

    def calculate_DTW_similarities(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Calculate DTW-based similarity between candidates and templates.

        DTW (Dynamic Time Warping) measures similarity between temporal sequences that may vary in speed.
        We convert to similarity by using 1/(1+distance) so higher values mean more similar.

        Returns:
            Array of shape (n_candidates, n_templates) with similarity scores
        """
        if self.normalize_features and self.is_fitted:
            candidate_matrix = np.array(candidate_features)
            print(f"Original candidate_matrix shape: {candidate_matrix.shape}")
            # Reshape to 2D: (n_events, n_features)
            if candidate_matrix.ndim > 2:
                candidate_matrix = candidate_matrix.reshape(candidate_matrix.shape[1], -1) # original shape (n_candidates, n_events, n_features)
            print(f"Reshaped candidate_matrix shape: {candidate_matrix.shape}")
            candidate_features = self.scaler.transform(candidate_matrix)
        print("Candidate features normalized using StandardScaler\n")
        
        similarities = np.zeros((len(candidate_features), len(self.templates)))
        
        for i, template in enumerate(self.templates):
            
            # Use DTW distance for time-series alignment
            distances = np.array([
                dtw.distance(candidate, template)
                for candidate in candidate_features
            ])
            
            # Convert distance to similarity: smaller distance = higher similarity
            similarities[:, i] = 1 / (1 + distances)
        
        return similarities
    
    def calculate_wasserstein_similarities(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Calculate Wasserstein distance-based similarity between candidates and templates.

        Wasserstein distance measures the cost of transforming one distribution into another.
        We convert to similarity by using 1/(1+distance) so higher values mean more similar.

        Returns:
            Array of shape (n_candidates, n_templates) with similarity scores
        """
        if self.normalize_features and self.is_fitted:
            candidate_matrix = np.array(candidate_features)
            print(f"Original candidate_matrix shape: {candidate_matrix.shape}")
            # Reshape to 2D: (n_events, n_features)
            if candidate_matrix.ndim > 2:
                candidate_matrix = candidate_matrix.reshape(candidate_matrix.shape[1], -1) # original shape (n_candidates, n_events, n_features)
            print(f"Reshaped candidate_matrix shape: {candidate_matrix.shape}")
            candidate_features = self.scaler.transform(candidate_matrix)
        print("Candidate features normalized using StandardScaler\n")
        
        similarities = np.zeros((len(candidate_features), len(self.templates)))
        
        for i, template in enumerate(self.templates):
            # Use Wasserstein distance
            distances = np.array([
                wasserstein_distance(candidate, template)
                for candidate in candidate_features
            ])
            
            # Convert distance to similarity: smaller distance = higher similarity
            similarities[:, i] = 1 / (1 + distances)
        
        return similarities