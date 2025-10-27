"""Evaluation framework for ML models"""

import numpy as np
from typing import Dict, List, Tuple
import time


class ModelEvaluator:
    """Evaluate ML model performance"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_prediction_horizon(self, 
                                   predictions: Dict[str, List[float]],
                                   ground_truth: Dict[str, List[float]]) -> Dict[str, float]:
        """Evaluate prediction accuracy across horizons"""
        accuracies = {}
        
        horizons = [1, 5, 15, 30, 60]
        
        for horizon in horizons:
            if str(horizon) in predictions and str(horizon) in ground_truth:
                pred = predictions[str(horizon)]
                truth = ground_truth[str(horizon)]
                
                mae = np.mean(np.abs(np.array(pred) - np.array(truth)))
                accuracy = 1 - (mae / (np.mean(truth) + 1e-10))
                
                accuracies[f'horizon_{horizon}s'] = accuracy
        
        return accuracies
    
    def evaluate_response_time(self, timings: List[float]) -> Dict[str, float]:
        """Evaluate response time metrics"""
        return {
            'mean_response_time': np.mean(timings),
            'p50_response_time': np.percentile(timings, 50),
            'p95_response_time': np.percentile(timings, 95),
            'p99_response_time': np.percentile(timings, 99),
        }
    
    def evaluate_detection_accuracy(self, 
                                   predictions: List[str],
                                   ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate classification accuracy"""
        correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
        total = len(predictions)
        
        accuracy = correct / total if total > 0 else 0
        
        true_positives = sum(1 for p, t in zip(predictions, ground_truth) 
                           if p == 'elephant' and t == 'elephant')
        false_positives = sum(1 for p, t in zip(predictions, ground_truth) 
                            if p == 'elephant' and t == 'mice')
        false_negatives = sum(1 for p, t in zip(predictions, ground_truth) 
                            if p == 'mice' and t == 'elephant')
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
    
    def generate_report(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("ML Model Evaluation Report")
        report.append("=" * 60)
        
        for metric_type, values in metrics.items():
            report.append(f"\n{metric_type}:")
            report.append("-" * 40)
            for key, value in values.items():
                report.append(f"  {key}: {value:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
