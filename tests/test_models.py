"""Tests for ML models"""

import pytest
import numpy as np
from sdn_ml.models import ElephantFlowDetector, LSTMTrafficPredictor, AnomalyDetector


def test_elephant_detector():
    """Test elephant flow detector"""
    detector = ElephantFlowDetector()
    
    features = {
        'flow_duration': 15.0,
        'total_bytes': 1e9,
        'bytes_per_packet_mean': 1500,
        'instantaneous_throughput': 800,
        'average_throughput': 850,
        'retransmission_rate': 0.01,
        'application_type': 5,
    }
    
    prob, classification = detector.predict(features)
    
    assert isinstance(prob, float)
    assert classification in ['elephant', 'mice', 'monitor']


def test_lstm_predictor():
    """Test LSTM predictor"""
    predictor = LSTMTrafficPredictor()
    
    history = [[np.random.random(25) for _ in range(60)]]
    
    predictions, confidence = predictor.predict(history)
    
    assert len(predictions) == 5
    assert isinstance(confidence, float)


def test_anomaly_detector():
    """Test anomaly detector"""
    detector = AnomalyDetector()
    
    features = {
        'network_load': 0.85,
        'hotspot_count': 5,
        'controller_cpu_usage': 80,
        'packet_rate': 50000,
        'drop_rate': 0.15,
        'new_flow_rate': 10000,
        'source_entropy': 0.1,
        'packet_size_distribution': 0.05,
    }
    
    score, classification = detector.detect(features)
    
    assert isinstance(score, float)
    assert classification in ['normal', 'suspicious', 'anomalous']
