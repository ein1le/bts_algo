"""
Visibility Graph Utilities for Time Series Analysis.

This module provides functions to convert time series data into visibility graphs
using various algorithms including Horizontal Visibility Graph (HVG), Natural 
Visibility Graph (NVG), and Divide and Conquer (DC) algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from scipy.sparse import csr_matrix
import logging

try:
    from ts2vg import HorizontalVG, NaturalVG
    TS2VG_AVAILABLE = True
except ImportError:
    TS2VG_AVAILABLE = False
    logging.warning("ts2vg package not available. Using custom implementations.")

logger = logging.getLogger(__name__)


class VisibilityGraphConverter:
    """
    Converter for transforming time series into visibility graphs.
    
    Supports multiple visibility graph algorithms:
    - Natural Visibility Graph (NVG)
    - Horizontal Visibility Graph (HVG) 
    - Limited Penetrable Visibility Graph (LPVG)
    - Divide and Conquer optimized versions
    """
    
    def __init__(self, directed: bool = False, weighted: bool = False):
        """
        Initialize visibility graph converter.
        
        Args:
            directed: Whether to create directed graphs
            weighted: Whether to add edge weights based on visibility strength
        """
        self.directed = directed
        self.weighted = weighted
        
    def natural_visibility_graph(self, ts: np.ndarray, use_ts2vg: bool = True) -> nx.Graph:
        """
        Create Natural Visibility Graph from time series.
        
        In NVG, two data points can see each other if all intermediate points
        are below the line connecting them.
        
        Args:
            ts: Time series data
            use_ts2vg: Whether to use ts2vg library if available
            
        Returns:
            NetworkX graph representing the visibility graph
        """
        if use_ts2vg and TS2VG_AVAILABLE:
            return self._natural_vg_ts2vg(ts)
        else:
            return self._natural_vg_custom(ts)
    
    def horizontal_visibility_graph(self, ts: np.ndarray, use_ts2vg: bool = True) -> nx.Graph:
        """
        Create Horizontal Visibility Graph from time series.
        
        In HVG, two data points can see each other if all intermediate points
        are below the minimum of the two points (horizontal line).
        
        Args:
            ts: Time series data
            use_ts2vg: Whether to use ts2vg library if available
            
        Returns:
            NetworkX graph representing the visibility graph
        """
        if use_ts2vg and TS2VG_AVAILABLE:
            return self._horizontal_vg_ts2vg(ts)
        else:
            return self._horizontal_vg_custom(ts)
    
    def limited_penetrable_vg(self, ts: np.ndarray, penetrable_limit: int = 1) -> nx.Graph:
        """
        Create Limited Penetrable Visibility Graph.
        
        Allows limited penetration through intermediate points.
        
        Args:
            ts: Time series data
            penetrable_limit: Maximum number of penetrations allowed
            
        Returns:
            NetworkX graph
        """
        return self._limited_penetrable_vg_custom(ts, penetrable_limit)
    
    def _natural_vg_ts2vg(self, ts: np.ndarray) -> nx.Graph:
        """Natural VG using ts2vg library."""
        vg = NaturalVG(directed=self.directed)
        vg.build(ts)
        
        # Convert to NetworkX
        edges = vg.edges
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(len(ts)))
        
        # Add edges
        for edge in edges:
            i, j = edge
            if self.weighted:
                weight = self._calculate_visibility_weight(ts, i, j)
                G.add_edge(i, j, weight=weight)
            else:
                G.add_edge(i, j)
        
        return G
    
    def _horizontal_vg_ts2vg(self, ts: np.ndarray) -> nx.Graph:
        """Horizontal VG using ts2vg library."""
        vg = HorizontalVG(directed=self.directed)
        vg.build(ts)
        
        # Convert to NetworkX
        edges = vg.edges
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(len(ts)))
        
        # Add edges
        for edge in edges:
            i, j = edge
            if self.weighted:
                weight = self._calculate_visibility_weight(ts, i, j)
                G.add_edge(i, j, weight=weight)
            else:
                G.add_edge(i, j)
        
        return G
    
    def _natural_vg_custom(self, ts: np.ndarray) -> nx.Graph:
        """Custom implementation of Natural Visibility Graph."""
        n = len(ts)
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n))
        
        # Check visibility between all pairs
        for i in range(n):
            for j in range(i + 2, n):  # Skip adjacent nodes (always connected)
                if self._has_natural_visibility(ts, i, j):
                    if self.weighted:
                        weight = self._calculate_visibility_weight(ts, i, j)
                        G.add_edge(i, j, weight=weight)
                    else:
                        G.add_edge(i, j)
        
        # Add edges between consecutive nodes
        for i in range(n - 1):
            if self.weighted:
                weight = self._calculate_visibility_weight(ts, i, i + 1)
                G.add_edge(i, i + 1, weight=weight)
            else:
                G.add_edge(i, i + 1)
        
        return G
    
    def _horizontal_vg_custom(self, ts: np.ndarray) -> nx.Graph:
        """Custom implementation of Horizontal Visibility Graph."""
        n = len(ts)
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n))
        
        # Check horizontal visibility between all pairs
        for i in range(n):
            for j in range(i + 2, n):  # Skip adjacent nodes
                if self._has_horizontal_visibility(ts, i, j):
                    if self.weighted:
                        weight = self._calculate_visibility_weight(ts, i, j)
                        G.add_edge(i, j, weight=weight)
                    else:
                        G.add_edge(i, j)
        
        # Add edges between consecutive nodes
        for i in range(n - 1):
            if self.weighted:
                weight = self._calculate_visibility_weight(ts, i, i + 1)
                G.add_edge(i, i + 1, weight=weight)
            else:
                G.add_edge(i, i + 1)
        
        return G
    
    def _limited_penetrable_vg_custom(self, ts: np.ndarray, penetrable_limit: int) -> nx.Graph:
        """Custom implementation of Limited Penetrable Visibility Graph."""
        n = len(ts)
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n))
        
        # Check penetrable visibility between all pairs
        for i in range(n):
            for j in range(i + 2, n):
                if self._has_penetrable_visibility(ts, i, j, penetrable_limit):
                    if self.weighted:
                        weight = self._calculate_visibility_weight(ts, i, j)
                        G.add_edge(i, j, weight=weight)
                    else:
                        G.add_edge(i, j)
        
        # Add edges between consecutive nodes
        for i in range(n - 1):
            if self.weighted:
                weight = self._calculate_visibility_weight(ts, i, i + 1)
                G.add_edge(i, i + 1, weight=weight)
            else:
                G.add_edge(i, i + 1)
        
        return G
    
    def _has_natural_visibility(self, ts: np.ndarray, i: int, j: int) -> bool:
        """Check if two points have natural visibility."""
        if i >= j:
            return False
        
        # Check if all intermediate points are below the line connecting i and j
        for k in range(i + 1, j):
            # Line equation: y = y_i + (y_j - y_i) * (x - x_i) / (x_j - x_i)
            line_value = ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i)
            if ts[k] >= line_value:
                return False
        
        return True
    
    def _has_horizontal_visibility(self, ts: np.ndarray, i: int, j: int) -> bool:
        """Check if two points have horizontal visibility."""
        if i >= j:
            return False
        
        # Check if all intermediate points are below min(ts[i], ts[j])
        threshold = min(ts[i], ts[j])
        for k in range(i + 1, j):
            if ts[k] >= threshold:
                return False
        
        return True
    
    def _has_penetrable_visibility(self, ts: np.ndarray, i: int, j: int, 
                                 penetrable_limit: int) -> bool:
        """Check if two points have penetrable visibility."""
        if i >= j:
            return False
        
        # Count penetrations (points above the connecting line)
        penetrations = 0
        for k in range(i + 1, j):
            line_value = ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i)
            if ts[k] >= line_value:
                penetrations += 1
                if penetrations > penetrable_limit:
                    return False
        
        return True
    
    def _calculate_visibility_weight(self, ts: np.ndarray, i: int, j: int) -> float:
        """Calculate edge weight based on visibility strength."""
        # Weight based on inverse distance and value similarity
        distance_weight = 1.0 / (abs(j - i) + 1)
        value_similarity = 1.0 / (abs(ts[i] - ts[j]) + 1e-8)
        
        return distance_weight * value_similarity


class DivideConquerVisibilityGraph:
    """
    Divide and Conquer algorithm for efficient visibility graph construction.
    
    Optimizes the O(n²) complexity of standard visibility graph algorithms
    by using divide and conquer approach.
    """
    
    def __init__(self, algorithm: str = 'natural'):
        """
        Initialize DC visibility graph constructor.
        
        Args:
            algorithm: 'natural' or 'horizontal'
        """
        self.algorithm = algorithm
        
    def build_graph(self, ts: np.ndarray) -> nx.Graph:
        """
        Build visibility graph using divide and conquer.
        
        Args:
            ts: Time series data
            
        Returns:
            NetworkX graph
        """
        n = len(ts)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add consecutive edges
        for i in range(n - 1):
            G.add_edge(i, i + 1)
        
        # Divide and conquer for non-consecutive edges
        edges = self._dc_visibility(ts, 0, n - 1)
        G.add_edges_from(edges)
        
        return G
    
    def _dc_visibility(self, ts: np.ndarray, left: int, right: int) -> List[Tuple[int, int]]:
        """Recursive divide and conquer visibility computation."""
        if right - left <= 1:
            return []
        
        edges = []
        mid = (left + right) // 2
        
        # Recursively process left and right halves
        edges.extend(self._dc_visibility(ts, left, mid))
        edges.extend(self._dc_visibility(ts, mid, right))
        
        # Find cross-partition edges
        edges.extend(self._cross_partition_edges(ts, left, mid, right))
        
        return edges
    
    def _cross_partition_edges(self, ts: np.ndarray, left: int, mid: int, 
                             right: int) -> List[Tuple[int, int]]:
        """Find visibility edges crossing the partition."""
        edges = []
        
        for i in range(left, mid + 1):
            for j in range(mid + 1, right + 1):
                if self.algorithm == 'natural':
                    if self._has_natural_visibility_range(ts, i, j):
                        edges.append((i, j))
                elif self.algorithm == 'horizontal':
                    if self._has_horizontal_visibility_range(ts, i, j):
                        edges.append((i, j))
        
        return edges
    
    def _has_natural_visibility_range(self, ts: np.ndarray, i: int, j: int) -> bool:
        """Check natural visibility for specific range."""
        for k in range(i + 1, j):
            line_value = ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i)
            if ts[k] >= line_value:
                return False
        return True
    
    def _has_horizontal_visibility_range(self, ts: np.ndarray, i: int, j: int) -> bool:
        """Check horizontal visibility for specific range."""
        threshold = min(ts[i], ts[j])
        for k in range(i + 1, j):
            if ts[k] >= threshold:
                return False
        return True


class GraphFeatureExtractor:
    """Extract graph-based features from visibility graphs."""
    
    @staticmethod
    def extract_node_features(G: nx.Graph, ts: np.ndarray) -> np.ndarray:
        """
        Extract node-level features from visibility graph.
        
        Args:
            G: Visibility graph
            ts: Original time series
            
        Returns:
            Node feature matrix [n_nodes, n_features]
        """
        n_nodes = len(G.nodes())
        features = np.zeros((n_nodes, 8))  # 8 features per node
        
        # Calculate graph metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        clustering = nx.clustering(G)
        
        for i, node in enumerate(G.nodes()):
            features[i, 0] = ts[node]  # Original time series value
            features[i, 1] = G.degree(node)  # Node degree
            features[i, 2] = degree_centrality[node]
            features[i, 3] = betweenness_centrality[node]
            features[i, 4] = closeness_centrality[node]
            features[i, 5] = eigenvector_centrality[node]
            features[i, 6] = clustering[node]
            features[i, 7] = node  # Node position (temporal information)
        
        return features
    
    @staticmethod
    def extract_graph_features(G: nx.Graph) -> Dict[str, float]:
        """
        Extract graph-level features.
        
        Args:
            G: Visibility graph
            
        Returns:
            Dictionary of graph features
        """
        features = {}
        
        # Basic graph properties
        features['num_nodes'] = G.number_of_nodes()
        features['num_edges'] = G.number_of_edges()
        features['density'] = nx.density(G)
        features['transitivity'] = nx.transitivity(G)
        features['avg_clustering'] = nx.average_clustering(G)
        
        # Path-based features
        if nx.is_connected(G):
            features['avg_path_length'] = nx.average_shortest_path_length(G)
            features['diameter'] = nx.diameter(G)
            features['radius'] = nx.radius(G)
        else:
            # For disconnected graphs
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            features['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            features['diameter'] = nx.diameter(subgraph)
            features['radius'] = nx.radius(subgraph)
        
        # Degree distribution features
        degrees = [G.degree(n) for n in G.nodes()]
        features['avg_degree'] = np.mean(degrees)
        features['degree_variance'] = np.var(degrees)
        features['max_degree'] = max(degrees)
        features['min_degree'] = min(degrees)
        
        # Small world properties
        try:
            # Random graph comparison
            random_clustering = 1.0 / features['num_nodes']  # Approximation
            random_path_length = np.log(features['num_nodes'])  # Approximation
            
            features['small_world_coefficient'] = (
                (features['avg_clustering'] / random_clustering) /
                (features['avg_path_length'] / random_path_length)
            )
        except:
            features['small_world_coefficient'] = 0.0
        
        return features
    
    @staticmethod
    def get_adjacency_matrix(G: nx.Graph, weighted: bool = False) -> np.ndarray:
        """
        Get adjacency matrix representation.
        
        Args:
            G: NetworkX graph
            weighted: Whether to include edge weights
            
        Returns:
            Adjacency matrix
        """
        if weighted and nx.is_weighted(G):
            return nx.adjacency_matrix(G, weight='weight').todense()
        else:
            return nx.adjacency_matrix(G).todense()


def convert_multivariate_to_graphs(data: np.ndarray, 
                                 method: str = 'natural',
                                 window_size: Optional[int] = None) -> List[nx.Graph]:
    """
    Convert multivariate time series to visibility graphs.
    
    Args:
        data: Multivariate time series [n_samples, n_features] or [n_features, n_samples]
        method: Visibility graph method ('natural', 'horizontal', 'lpvg')
        window_size: If provided, create graphs from sliding windows
        
    Returns:
        List of visibility graphs (one per feature or window)
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    
    # Assume features are in columns (standard format)
    if data.shape[0] > data.shape[1]:
        # More samples than features - transpose
        data = data.T
    
    converter = VisibilityGraphConverter(weighted=True)
    graphs = []
    
    if window_size is None:
        # One graph per feature
        for feature_idx in range(data.shape[0]):
            ts = data[feature_idx, :]
            
            if method == 'natural':
                graph = converter.natural_visibility_graph(ts)
            elif method == 'horizontal':
                graph = converter.horizontal_visibility_graph(ts)
            elif method == 'lpvg':
                graph = converter.limited_penetrable_vg(ts)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            graphs.append(graph)
    else:
        # Sliding window approach
        n_features, n_samples = data.shape
        
        for start_idx in range(n_samples - window_size + 1):
            window_data = data[:, start_idx:start_idx + window_size]
            
            # Create graph from concatenated window (or use first feature)
            ts = window_data[0, :]  # Use first feature for simplicity
            
            if method == 'natural':
                graph = converter.natural_visibility_graph(ts)
            elif method == 'horizontal':
                graph = converter.horizontal_visibility_graph(ts)
            elif method == 'lpvg':
                graph = converter.limited_penetrable_vg(ts)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            graphs.append(graph)
    
    return graphs


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    
    # Generate sample time series
    t = np.linspace(0, 4 * np.pi, 100)
    ts = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Test different visibility graph methods
    converter = VisibilityGraphConverter(weighted=True)
    
    # Natural VG
    nvg = converter.natural_visibility_graph(ts, use_ts2vg=False)
    print(f"Natural VG: {nvg.number_of_nodes()} nodes, {nvg.number_of_edges()} edges")
    
    # Horizontal VG
    hvg = converter.horizontal_visibility_graph(ts, use_ts2vg=False)
    print(f"Horizontal VG: {hvg.number_of_nodes()} nodes, {hvg.number_of_edges()} edges")
    
    # Limited Penetrable VG
    lpvg = converter.limited_penetrable_vg(ts, penetrable_limit=1)
    print(f"Limited Penetrable VG: {lpvg.number_of_nodes()} nodes, {lpvg.number_of_edges()} edges")
    
    # Extract features
    extractor = GraphFeatureExtractor()
    node_features = extractor.extract_node_features(nvg, ts)
    graph_features = extractor.extract_graph_features(nvg)
    
    print(f"Node features shape: {node_features.shape}")
    print(f"Graph features: {len(graph_features)} features")
    print("Sample graph features:", list(graph_features.keys())[:5]) 