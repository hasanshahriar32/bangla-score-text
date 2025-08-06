"""
Plagiarism Detection Model
Combines similarity scoring with plagiarism analysis logic
"""

from typing import List, Optional
import logging
from datetime import datetime

from models.similarity_model import SimilarityModel
from schemas.response_models import PlagiarismResponse, SimilarityResult, PlagiarismAnalysis
from utils.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class PlagiarismDetector:
    """
    Main plagiarism detection engine that combines similarity scoring with plagiarism analysis
    """
    
    def __init__(self, model_name: str = "tfidf-bangla"):
        """
        Initialize plagiarism detector
        
        Args:
            model_name: Sentence transformer model name for similarity calculation
        """
        self.similarity_model = SimilarityModel(model_name)
        self.text_preprocessor = TextPreprocessor()
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the plagiarism detector by loading the ML model"""
        try:
            await self.similarity_model.load_model()
            self.is_initialized = True
            logger.info("Plagiarism detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize plagiarism detector: {str(e)}")
            raise
    
    async def detect_plagiarism(
        self,
        target_text: str,
        candidate_texts: List[str],
        threshold: Optional[float] = None,
        include_preprocessing: bool = True
    ) -> PlagiarismResponse:
        """
        Perform comprehensive plagiarism detection
        
        Args:
            target_text: Text to check for plagiarism
            candidate_texts: List of texts to compare against
            threshold: Similarity threshold for plagiarism detection (default: 0.7)
            include_preprocessing: Whether to preprocess texts before analysis
            
        Returns:
            PlagiarismResponse with detailed analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Plagiarism detector not initialized")
        
        # Set default threshold
        if threshold is None:
            threshold = 0.7
        
        try:
            # Preprocess texts if requested
            processed_target = target_text
            processed_candidates = candidate_texts
            
            if include_preprocessing:
                processed_target = self.text_preprocessor.preprocess(target_text)
                processed_candidates = [
                    self.text_preprocessor.preprocess(text) for text in candidate_texts
                ]
            
            # Calculate similarity scores
            similarity_results = await self.calculate_similarities(
                target_text=processed_target,
                candidate_texts=processed_candidates,
                include_preprocessing=False  # Already preprocessed
            )
            
            # Analyze plagiarism
            plagiarism_analysis = self._analyze_plagiarism(similarity_results, threshold)
            
            # Create response
            response = PlagiarismResponse(
                target_text=target_text,
                similarity_results=similarity_results,
                plagiarism_analysis=plagiarism_analysis,
                threshold_used=threshold,
                has_plagiarism=plagiarism_analysis.potential_plagiarism,
                max_similarity_score=plagiarism_analysis.max_similarity_score,
                timestamp=datetime.utcnow(),
                preprocessing_applied=include_preprocessing
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in plagiarism detection: {str(e)}")
            raise
    
    async def calculate_similarities(
        self,
        target_text: str,
        candidate_texts: List[str],
        include_preprocessing: bool = True
    ) -> List[SimilarityResult]:
        """
        Calculate similarity scores between target text and candidates
        
        Args:
            target_text: Target text for comparison
            candidate_texts: List of candidate texts
            include_preprocessing: Whether to preprocess texts
            
        Returns:
            List of SimilarityResult objects
        """
        if not self.is_initialized:
            raise RuntimeError("Plagiarism detector not initialized")
        
        try:
            # Preprocess texts if requested
            processed_target = target_text
            processed_candidates = candidate_texts
            
            if include_preprocessing:
                processed_target = self.text_preprocessor.preprocess(target_text)
                processed_candidates = [
                    self.text_preprocessor.preprocess(text) for text in candidate_texts
                ]
            
            # Get similarity metrics from the model
            metrics = self.similarity_model.get_similarity_metrics(
                processed_target, processed_candidates
            )
            
            # Create SimilarityResult objects
            similarity_results = []
            for i, (candidate_text, similarity_score, additional_metrics) in enumerate(metrics):
                result = SimilarityResult(
                    candidate_index=i,
                    candidate_text=candidate_texts[i],  # Use original text
                    similarity_score=similarity_score,
                    confidence=additional_metrics["confidence"],
                    length_ratio=additional_metrics["length_ratio"],
                    target_length=additional_metrics["target_length"],
                    candidate_length=len(candidate_texts[i])  # Original length
                )
                similarity_results.append(result)
            
            # Sort by similarity score (descending)
            similarity_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            raise
    
    def _analyze_plagiarism(
        self,
        similarity_results: List[SimilarityResult],
        threshold: float
    ) -> PlagiarismAnalysis:
        """
        Analyze similarity results to determine plagiarism likelihood
        
        Args:
            similarity_results: List of similarity results
            threshold: Similarity threshold for plagiarism detection
            
        Returns:
            PlagiarismAnalysis with detailed analysis
        """
        try:
            # Find matches above threshold
            potential_matches = [
                result for result in similarity_results 
                if result.similarity_score >= threshold
            ]
            
            # Calculate statistics
            max_similarity = max(
                [result.similarity_score for result in similarity_results],
                default=0.0
            )
            
            avg_similarity = sum(
                [result.similarity_score for result in similarity_results]
            ) / len(similarity_results) if similarity_results else 0.0
            
            # Determine plagiarism risk level
            risk_level = self._calculate_risk_level(max_similarity, len(potential_matches), threshold)
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(
                potential_matches, max_similarity, avg_similarity, threshold, risk_level
            )
            
            analysis = PlagiarismAnalysis(
                potential_plagiarism=len(potential_matches) > 0,
                matches_above_threshold=len(potential_matches),
                max_similarity_score=max_similarity,
                average_similarity_score=avg_similarity,
                risk_level=risk_level,
                matched_candidate_indices=[match.candidate_index for match in potential_matches],
                analysis_summary=summary
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in plagiarism analysis: {str(e)}")
            raise
    
    def _calculate_risk_level(
        self,
        max_similarity: float,
        num_matches: int,
        threshold: float
    ) -> str:
        """
        Calculate risk level based on similarity scores and number of matches
        
        Args:
            max_similarity: Maximum similarity score found
            num_matches: Number of matches above threshold
            threshold: Similarity threshold used
            
        Returns:
            Risk level string: "low", "medium", "high", or "critical"
        """
        if max_similarity >= 0.9:
            return "critical"
        elif max_similarity >= 0.8:
            return "high"
        elif max_similarity >= threshold:
            return "medium" if num_matches <= 2 else "high"
        else:
            return "low"
    
    def _generate_analysis_summary(
        self,
        potential_matches: List[SimilarityResult],
        max_similarity: float,
        avg_similarity: float,
        threshold: float,
        risk_level: str
    ) -> str:
        """
        Generate human-readable analysis summary
        
        Args:
            potential_matches: List of potential plagiarism matches
            max_similarity: Maximum similarity score
            avg_similarity: Average similarity score
            threshold: Threshold used
            risk_level: Calculated risk level
            
        Returns:
            Analysis summary string
        """
        summary_parts = []
        
        if not potential_matches:
            summary_parts.append(
                f"No plagiarism detected. Maximum similarity score of {max_similarity:.3f} "
                f"is below the threshold of {threshold:.3f}."
            )
        else:
            summary_parts.append(
                f"Potential plagiarism detected with {len(potential_matches)} matches above threshold. "
                f"Risk level: {risk_level.upper()}."
            )
            
            summary_parts.append(
                f"Maximum similarity score: {max_similarity:.3f}, "
                f"Average similarity: {avg_similarity:.3f}."
            )
            
            if max_similarity >= 0.9:
                summary_parts.append(
                    "CRITICAL: Very high similarity detected. Text appears to be nearly identical to source material."
                )
            elif max_similarity >= 0.8:
                summary_parts.append(
                    "HIGH RISK: Significant similarity detected. Manual review recommended."
                )
            elif len(potential_matches) > 3:
                summary_parts.append(
                    "Multiple potential sources detected. Consider reviewing text originality."
                )
        
        return " ".join(summary_parts)
