"""
Similarity Model for Bangla Text Processing
Uses TF-IDF vectorization with cosine similarity for text similarity calculation
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SimilarityModel:
    """
    Handles text similarity calculations using TF-IDF vectorization
    Optimized for Bangla language text processing
    """
    
    def __init__(self, model_name: str = "tfidf-bangla"):
        """
        Initialize the similarity model
        
        Args:
            model_name: Name of the model to use (for compatibility)
        """
        self.model_name = model_name
        self.vectorizer = None
        self.is_loaded = False
    
    async def load_model(self):
        """Load the TF-IDF vectorizer"""
        try:
            logger.info(f"Initializing TF-IDF vectorizer for text similarity")
            # Create TF-IDF vectorizer optimized for multilingual text
            self.vectorizer = TfidfVectorizer(
                analyzer='char_wb',  # Character n-grams work well for non-English text
                ngram_range=(2, 5),  # Use 2-5 character n-grams
                max_features=10000,  # Limit features for performance
                min_df=1,           # Include even rare terms
                max_df=0.95,        # Ignore very common terms
                lowercase=True,     # Convert to lowercase
                strip_accents=None  # Don't strip accents for Bangla
            )
            self.is_loaded = True
            logger.info("TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into TF-IDF vectors
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of TF-IDF vectors
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Fit and transform the texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors
        
        Args:
            embedding1: First text TF-IDF vector
            embedding2: Second text TF-IDF vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Reshape if needed for cosine_similarity function
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # Ensure similarity is between 0 and 1 (cosine similarity is already 0-1 for TF-IDF)
            return max(0.0, float(similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def batch_similarity(self, target_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> List[float]:
        """
        Calculate similarity between target and multiple candidates
        
        Args:
            target_embedding: Target text TF-IDF vector
            candidate_embeddings: Array of candidate text TF-IDF vectors
            
        Returns:
            List of similarity scores
        """
        try:
            if target_embedding.ndim == 1:
                target_embedding = target_embedding.reshape(1, -1)
            
            similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
            
            # Ensure similarities are between 0 and 1
            normalized_similarities = [max(0.0, float(sim)) for sim in similarities]
            
            return normalized_similarities
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {str(e)}")
            raise
    
    def get_similarity_metrics(self, target_text: str, candidate_texts: List[str]) -> List[Tuple[str, float, dict]]:
        """
        Get detailed similarity metrics for target text against candidates
        
        Args:
            target_text: Target text for comparison
            candidate_texts: List of candidate texts
            
        Returns:
            List of tuples containing (candidate_text, similarity_score, additional_metrics)
        """
        try:
            # Encode all texts
            all_texts = [target_text] + candidate_texts
            embeddings = self.encode_texts(all_texts)
            
            target_embedding = embeddings[0:1]  # Keep 2D shape
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = self.batch_similarity(target_embedding, candidate_embeddings)
            
            # Calculate additional metrics
            results = []
            for i, (candidate_text, similarity) in enumerate(zip(candidate_texts, similarities)):
                # Calculate text length ratio
                length_ratio = len(candidate_text) / len(target_text) if len(target_text) > 0 else 0
                
                # Calculate confidence based on text length and similarity
                confidence = self._calculate_confidence(similarity, length_ratio)
                
                additional_metrics = {
                    "confidence": confidence,
                    "length_ratio": length_ratio,
                    "target_length": len(target_text),
                    "candidate_length": len(candidate_text)
                }
                
                results.append((candidate_text, similarity, additional_metrics))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similarity metrics: {str(e)}")
            raise
    
    def _calculate_confidence(self, similarity: float, length_ratio: float) -> float:
        """
        Calculate confidence score based on similarity and text length ratio
        
        Args:
            similarity: Similarity score between 0 and 1
            length_ratio: Ratio of candidate text length to target text length
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence is the similarity score
        base_confidence = similarity
        
        # Adjust confidence based on length ratio
        # Penalize very short or very long texts compared to target
        if length_ratio < 0.3 or length_ratio > 3.0:
            length_penalty = 0.2
        elif length_ratio < 0.5 or length_ratio > 2.0:
            length_penalty = 0.1
        else:
            length_penalty = 0.0
        
        confidence = max(0.0, base_confidence - length_penalty)
        
        return confidence
