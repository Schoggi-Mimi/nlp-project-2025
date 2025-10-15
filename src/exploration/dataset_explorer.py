"""Dataset exploration module for computing basic statistics and analysis."""
import string
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from langdetect import LangDetectException, detect, detect_langs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    stops = set(stopwords.words('english'))
    punct = set(string.punctuation)
except LookupError:
    nltk.download('stopwords', quiet=True)
    stops = set(stopwords.words('english'))
    punct = set(string.punctuation)


class DatasetExplorer:
    """Explore and analyze text datasets."""
    
    def __init__(self, data: pd.DataFrame, text_column: str = 'text', 
                 label_column: str = None):
        """
        Initialize the DatasetExplorer.
        
        Args:
            data: DataFrame containing the dataset
            text_column: Name of the column containing text
            label_column: Name of the column containing labels (optional)
        """
        self.data = data
        self.text_column = text_column
        self.label_column = label_column
        
    def compute_basic_stats(self) -> Dict:
        """
        Compute basic statistics about the dataset.
        
        Returns:
            Dictionary containing basic statistics
        """
        stats = {
            'total_samples': len(self.data),
            'missing_values': self.data[self.text_column].isna().sum(),
            'unique_samples': self.data[self.text_column].nunique()
        }
        
        # Add class balance if labels are provided
        if self.label_column and self.label_column in self.data.columns:
            class_counts = self.data[self.label_column].value_counts().to_dict()
            stats['class_distribution'] = class_counts
            stats['num_classes'] = len(class_counts)
            
            # Calculate class balance ratio (most common / least common)
            if len(class_counts) > 0:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                stats['class_balance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
        
        return stats
    
    def compute_term_frequency(self, top_n: int = 50) -> Counter:
        """
        Compute term frequency across the dataset.
        
        Args:
            top_n: Number of top terms to return
            
        Returns:
            Counter object with term frequencies
        """
        all_tokens = []
        filtered_tokens = []
        for text in self.data[self.text_column].dropna():
            try:
                tokens = word_tokenize(str(text).lower())
                all_tokens.extend(tokens)
                filtered_tokens.extend([t for t in tokens if t not in stops and t not in punct])

            except Exception as e:
                warnings.warn(f"Error tokenizing text: {e}")
                continue
        
        term_freq = Counter(all_tokens)
        filtered_term_freq = Counter(filtered_tokens)
        return term_freq.most_common(top_n), filtered_term_freq.most_common(top_n)

    def measure_text_length(self) -> Dict:
        """
        Measure text length statistics in both characters and tokens.
        
        Returns:
            Dictionary containing length statistics
        """
        char_lengths = []
        token_lengths = []
        
        for text in self.data[self.text_column].dropna():
            text_str = str(text)
            char_lengths.append(len(text_str))
            
            try:
                tokens = word_tokenize(text_str)
                token_lengths.append(len(tokens))
            except Exception:
                token_lengths.append(0)
        
        stats = {
            'char_length': {
                'min': int(np.min(char_lengths)) if char_lengths else 0,
                'max': int(np.max(char_lengths)) if char_lengths else 0,
                'avg': float(np.mean(char_lengths)) if char_lengths else 0.0,
                'median': float(np.median(char_lengths)) if char_lengths else 0.0,
                'std': float(np.std(char_lengths)) if char_lengths else 0.0
            },
            'token_length': {
                'min': int(np.min(token_lengths)) if token_lengths else 0,
                'max': int(np.max(token_lengths)) if token_lengths else 0,
                'avg': float(np.mean(token_lengths)) if token_lengths else 0.0,
                'median': float(np.median(token_lengths)) if token_lengths else 0.0,
                'std': float(np.std(token_lengths)) if token_lengths else 0.0
            }
        }
        
        return stats
    
    def detect_languages(self, sample_size: int = None) -> Dict:
        """
        Detect languages present in the dataset.
        
        Args:
            sample_size: Number of samples to check (None for all)
            
        Returns:
            Dictionary with language distribution
        """
        language_counts = Counter()
        
        # Use sample or all data
        data_to_check = self.data[self.text_column].dropna()
        if sample_size and sample_size < len(data_to_check):
            data_to_check = data_to_check.sample(n=sample_size, random_state=42)
        
        for text in data_to_check:
            try:
                text_str = str(text).strip()
                if len(text_str) > 0:
                    lang = detect(text_str)
                    language_counts[lang] += 1
            except LangDetectException:
                language_counts['unknown'] += 1
            except Exception:
                language_counts['error'] += 1
        
        total = sum(language_counts.values())
        language_dist = {
            lang: {
                'count': count,
                'percentage': round(100 * count / total, 2) if total > 0 else 0
            }
            for lang, count in language_counts.most_common()
        }
        non_en = self.data[self.text_column][self.data[self.text_column].apply(lambda t: detect(str(t)) != 'en')]
        non_en = non_en[non_en.notna()]
        return language_dist, non_en