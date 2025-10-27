"""Tests for feature extraction"""

import pytest
import numpy as np
from sdn_ml.features import FeatureExtractor


def test_feature_extractor():
    """Test feature extraction"""
    extractor = FeatureExtractor()
    
    flow_data = {
        'packets': [
            {'size': 1500, 'timestamp': i * 0.01}
            for i in range(100)
        ]
    }
    
    features = extractor.extract_flow_features(flow_data)
    
    assert 'flow_duration' in features
    assert 'total_bytes' in features
    assert 'total_packets' in features
    assert features['total_packets'] == 100


def test_time_window_features():
    """Test time window feature extraction"""
    extractor = FeatureExtractor()
    
    history = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    features = extractor.extract_time_window_features(history, 'throughput')
    
    assert len(features) > 0
    assert 'throughput_mean_5' in features


def test_fourier_features():
    """Test Fourier feature extraction"""
    extractor = FeatureExtractor()
    
    time_series = [np.sin(i) for i in range(100)]
    
    features = extractor.extract_fourier_features(time_series)
    
    assert 'dominant_frequency_1' in features
    assert 'dominant_frequency_2' in features
    assert 'dominant_frequency_3' in features
