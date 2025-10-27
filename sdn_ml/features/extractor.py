"""Feature extraction from Prometheus metrics"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats
from scipy.fft import fft


class FeatureExtractor:
    """Extract 40+ features from network metrics"""
    
    def __init__(self):
        self.windows = [5, 15, 30, 60, 300]
        self.history = {}
    
    def extract_flow_features(self, flow_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract flow-level features"""
        features = {}
        
        packets = flow_data.get('packets', [])
        if not packets:
            return self._empty_flow_features()
        
        packet_sizes = [p['size'] for p in packets]
        timestamps = [p['timestamp'] for p in packets]
        
        features['flow_duration'] = max(timestamps) - min(timestamps) if timestamps else 0
        features['total_bytes'] = sum(packet_sizes)
        features['total_packets'] = len(packets)
        features['bytes_per_packet_mean'] = np.mean(packet_sizes) if packet_sizes else 0
        features['bytes_per_packet_std'] = np.std(packet_sizes) if packet_sizes else 0
        
        if len(timestamps) > 1:
            inter_arrivals = np.diff(sorted(timestamps))
            features['inter_arrival_time'] = np.mean(inter_arrivals)
            features['inter_arrival_variance'] = np.std(inter_arrivals)
            features['burst_ratio'] = np.std(inter_arrivals) / (np.mean(inter_arrivals) + 1e-10)
        else:
            features['inter_arrival_time'] = 0
            features['inter_arrival_variance'] = 0
            features['burst_ratio'] = 0
        
        if features['flow_duration'] > 0:
            features['instantaneous_throughput'] = (features['total_bytes'] * 8) / features['flow_duration'] / 1e6
        else:
            features['instantaneous_throughput'] = 0
        
        return features
    
    def extract_port_features(self, port_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract port-level features"""
        features = {}
        
        features['port_utilization'] = port_data.get('utilization', 0)
        features['queue_depth'] = port_data.get('queue_depth', 0)
        features['drop_rate'] = port_data.get('drops', 0) / max(port_data.get('rx_packets', 1), 1)
        features['error_rate'] = port_data.get('errors', 0) / max(port_data.get('rx_packets', 1), 1)
        
        queue_history = port_data.get('queue_history', [])
        if queue_history:
            features['queue_depth_variance'] = np.var(queue_history)
            features['buffer_occupancy_time'] = sum(1 for q in queue_history if q > 0.8)
        else:
            features['queue_depth_variance'] = 0
            features['buffer_occupancy_time'] = 0
        
        return features
    
    def extract_path_features(self, path_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract path-level features"""
        features = {}
        
        features['end_to_end_latency'] = path_data.get('latency', 0)
        features['latency_variance'] = path_data.get('latency_jitter', 0)
        features['hop_count'] = path_data.get('hop_count', 0)
        features['bottleneck_bandwidth'] = path_data.get('bottleneck_bandwidth', 0)
        
        features['path_diversity_score'] = len(path_data.get('alternate_paths', []))
        features['disjoint_path_availability'] = 1 if features['path_diversity_score'] > 0 else 0
        
        return features
    
    def extract_network_features(self, network_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract network-wide features"""
        features = {}
        
        total_capacity = network_data.get('total_capacity', 1)
        total_traffic = network_data.get('total_traffic', 0)
        features['network_load'] = total_traffic / total_capacity if total_capacity > 0 else 0
        
        link_utilizations = network_data.get('link_utilizations', [])
        features['hotspot_count'] = sum(1 for u in link_utilizations if u > 0.8)
        
        if link_utilizations:
            utilization_dist = np.array(link_utilizations)
            features['load_distribution_entropy'] = stats.entropy(utilization_dist + 1e-10)
        else:
            features['load_distribution_entropy'] = 0
        
        features['link_failure_count'] = network_data.get('failed_links', 0)
        features['controller_cpu_usage'] = network_data.get('controller_cpu', 0)
        
        return features
    
    def extract_time_window_features(self, metric_history: List[float], metric_name: str) -> Dict[str, float]:
        """Extract time-window features"""
        features = {}
        
        if not metric_history:
            return {f'{metric_name}_mean_{w}': 0 for w in self.windows}
        
        for window in self.windows:
            window_data = metric_history[-window:] if len(metric_history) >= window else metric_history
            if window_data:
                features[f'{metric_name}_mean_{window}'] = np.mean(window_data)
                features[f'{metric_name}_std_{window}'] = np.std(window_data)
                features[f'{metric_name}_max_{window}'] = np.max(window_data)
                features[f'{metric_name}_trend_{window}'] = self._calculate_trend(window_data)
            else:
                features[f'{metric_name}_mean_{window}'] = 0
                features[f'{metric_name}_std_{window}'] = 0
                features[f'{metric_name}_max_{window}'] = 0
                features[f'{metric_name}_trend_{window}'] = 0
        
        return features
    
    def extract_fourier_features(self, time_series: List[float]) -> Dict[str, float]:
        """Extract Fourier transform features"""
        features = {}
        
        if len(time_series) < 3:
            return {'dominant_frequency_1': 0, 'dominant_frequency_2': 0, 'dominant_frequency_3': 0}
        
        fft_values = fft(time_series)
        frequencies = np.fft.fftfreq(len(time_series), d=1.0)
        
        magnitudes = np.abs(fft_values)
        top_indices = np.argsort(magnitudes)[-3:][::-1]
        
        features['dominant_frequency_1'] = abs(frequencies[top_indices[0]]) if len(top_indices) > 0 else 0
        features['dominant_frequency_2'] = abs(frequencies[top_indices[1]]) if len(top_indices) > 1 else 0
        features['dominant_frequency_3'] = abs(frequencies[top_indices[2]]) if len(top_indices) > 2 else 0
        
        return features
    
    def extract_wavelet_features(self, time_series: List[float]) -> Dict[str, float]:
        """Extract wavelet features"""
        features = {}
        
        if len(time_series) < 8:
            return {'wavelet_energy_level1': 0, 'wavelet_energy_level2': 0, 'wavelet_energy_level3': 0}
        
        try:
            # Simplified wavelet decomposition without pywt
            coeffs = [time_series[i::(2**(3-j))] for j in range(4)]
            features['wavelet_energy_level1'] = np.sum(np.square(coeffs[0]))
            features['wavelet_energy_level2'] = np.sum(np.square(coeffs[1])) if len(coeffs) > 1 else 0
            features['wavelet_energy_level3'] = np.sum(np.square(coeffs[2])) if len(coeffs) > 2 else 0
        except:
            features['wavelet_energy_level1'] = 0
            features['wavelet_energy_level2'] = 0
            features['wavelet_energy_level3'] = 0
        
        return features
    
    def extract_graph_features(self, topology: Any) -> Dict[str, float]:
        """Extract graph-based features"""
        features = {}
        
        import networkx as nx
        
        if isinstance(topology, dict):
            G = nx.Graph()
            for node, neighbors in topology.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
        else:
            G = topology
        
        if len(G.nodes()) == 0:
            return features
        
        centrality = nx.betweenness_centrality(G)
        avg_centrality = np.mean(list(centrality.values())) if centrality else 0
        features['avg_betweenness_centrality'] = avg_centrality
        
        clustering = nx.clustering(G)
        avg_clustering = np.mean(list(clustering.values())) if clustering else 0
        features['avg_clustering_coefficient'] = avg_clustering
        
        return features
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate linear regression slope"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _empty_flow_features(self) -> Dict[str, float]:
        """Return empty feature dict"""
        return {
            'flow_duration': 0,
            'total_bytes': 0,
            'total_packets': 0,
            'bytes_per_packet_mean': 0,
            'bytes_per_packet_std': 0,
            'inter_arrival_time': 0,
            'inter_arrival_variance': 0,
            'burst_ratio': 0,
            'instantaneous_throughput': 0,
        }
