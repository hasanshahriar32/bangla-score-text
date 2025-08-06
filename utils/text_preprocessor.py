"""
Text preprocessing utilities for Bangla language text
Handles text cleaning and normalization for better similarity calculation
"""

import re
import unicodedata
from typing import List

class TextPreprocessor:
    """
    Text preprocessing utilities optimized for Bangla language
    Handles cleaning, normalization, and preparation for ML model processing
    """
    
    def __init__(self):
        # Bangla Unicode range
        self.bangla_range = r'[\u0980-\u09FF]'
        
        # Common Bangla punctuation and symbols
        self.bangla_punctuation = r'[।॥‍ঃ]'
        
        # English punctuation
        self.english_punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
        
        # Numbers (both English and Bangla)
        self.numbers = r'[0-9০-৯]'
        
        # Whitespace patterns
        self.extra_whitespace = r'\s+'
    
    def preprocess(self, text: str) -> str:
        """
        Main preprocessing function that applies all cleaning steps
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Apply preprocessing steps
        processed_text = text
        processed_text = self.normalize_unicode(processed_text)
        processed_text = self.clean_text(processed_text)
        processed_text = self.normalize_whitespace(processed_text)
        processed_text = self.remove_extra_punctuation(processed_text)
        processed_text = processed_text.strip()
        
        return processed_text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to standard form
        
        Args:
            text: Input text
            
        Returns:
            Unicode normalized text
        """
        # Normalize to NFC form (canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        return normalized
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n+', ' ', text)
        
        # Replace tabs with spaces
        text = re.sub(r'\t+', ' ', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(self.extra_whitespace, ' ', text)
        
        return text
    
    def remove_extra_punctuation(self, text: str) -> str:
        """
        Remove or normalize excessive punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        # Replace multiple consecutive punctuation marks with single occurrence
        text = re.sub(r'([।॥‍ঃ!?.]){2,}', r'\1', text)
        
        # Remove excessive commas
        text = re.sub(r',{2,}', ',', text)
        
        return text
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove all numbers (both English and Bangla)
        
        Args:
            text: Input text
            
        Returns:
            Text without numbers
        """
        text = re.sub(self.numbers, '', text)
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove all punctuation marks
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        # Remove Bangla punctuation
        text = re.sub(self.bangla_punctuation, '', text)
        
        # Remove English punctuation
        text = re.sub(self.english_punctuation, '', text)
        
        return text
    
    def extract_bangla_text(self, text: str) -> str:
        """
        Extract only Bangla characters from text
        
        Args:
            text: Input text
            
        Returns:
            Text containing only Bangla characters and spaces
        """
        # Keep only Bangla characters and spaces
        bangla_text = re.sub(f'[^{self.bangla_range}\s]', '', text)
        
        # Normalize whitespace
        bangla_text = self.normalize_whitespace(bangla_text)
        
        return bangla_text.strip()
    
    def get_word_count(self, text: str) -> int:
        """
        Get word count for Bangla text
        
        Args:
            text: Input text
            
        Returns:
            Number of words
        """
        words = text.split()
        return len([word for word in words if word.strip()])
    
    def get_character_count(self, text: str, include_spaces: bool = True) -> int:
        """
        Get character count for text
        
        Args:
            text: Input text
            include_spaces: Whether to include spaces in count
            
        Returns:
            Number of characters
        """
        if include_spaces:
            return len(text)
        else:
            return len(text.replace(' ', ''))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split Bangla text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on Bangla sentence endings
        sentences = re.split(r'[।॥]+', text)
        
        # Also split on English sentence endings
        all_sentences = []
        for sentence in sentences:
            sub_sentences = re.split(r'[.!?]+', sentence)
            all_sentences.extend(sub_sentences)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in all_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def preprocess_for_similarity(self, text: str) -> str:
        """
        Specialized preprocessing for similarity calculation
        Applies aggressive cleaning for better similarity results
        
        Args:
            text: Input text
            
        Returns:
            Text optimized for similarity calculation
        """
        # Start with basic preprocessing
        processed = self.preprocess(text)
        
        # Convert to lowercase (for languages that support it)
        # Note: Bangla doesn't have case, but this helps with mixed content
        processed = processed.lower()
        
        # Remove excessive punctuation for similarity
        processed = self.remove_extra_punctuation(processed)
        
        # Normalize whitespace again
        processed = self.normalize_whitespace(processed)
        
        return processed.strip()
