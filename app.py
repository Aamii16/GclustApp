import os
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import kneighbors_graph, BallTree
from sklearn.metrics import pairwise_distances  # Add this import
from sklearn.neighbors import kneighbors_graph
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import os
from werkzeug.utils import secure_filename
import json
import requests
from datetime import datetime
import time
from collections import Counter
import re
import networkx as nx
import community as community_louvain  # python-louvain for Louvain algorithm
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'fallback-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['DEBUG'] = False  # Disable in production

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_content(file_path):
    """Basic file content validation"""
    try:
        # Check if file is actually an image
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                Image.open(file_path).verify()
                return True
            except Exception:
                return False
        
        # Check for CSV/JSON
        elif file_path.lower().endswith('.csv'):
            with open(file_path, 'r') as f:
                header = f.readline()
                if not header.strip() or ',' not in header:
                    return False
            return True
        
        elif file_path.lower().endswith('.json'):
            with open(file_path, 'r') as f:
                try:
                    json.load(f)
                    return True
                except json.JSONDecodeError:
                    return False
        
        return True
    except Exception:
        return False

def enhance_image_preprocessing(image_path):
    """Optimized image preprocessing with reduced resolution"""
    try:
        img = Image.open(image_path)
        original_img = img.copy()
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Reduced resolution for better performance
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Optional: Apply slight gaussian blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Optional: Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        img_array = np.array(img)
        original_array = np.array(original_img.resize((100, 100)))
        
        # Convert to different color spaces for better clustering
        pixels_rgb = img_array.reshape(-1, 3)
        
        # Add HSV features
        from colorsys import rgb_to_hsv
        hsv_pixels = np.array([rgb_to_hsv(r/255, g/255, b/255) for r, g, b in pixels_rgb])
        hsv_pixels = hsv_pixels * 255  # Scale back to 0-255
        
        # Combine RGB and HSV features
        enhanced_pixels = np.hstack([pixels_rgb, hsv_pixels])
        
        return enhanced_pixels, img_array.shape, original_array, pixels_rgb
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def improved_kmeans_clustering(data, n_clusters, random_state=42, **kwargs):
    """Improved K-means with multiple initializations and validation"""
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        max_iter = kwargs.get('max_iterations', 300)
        n_init = kwargs.get('n_init', 20)  # More initializations for better results
        
        # Try different initialization methods
        best_kmeans = None
        best_score = -1
        
        for init_method in ['k-means++', 'random']:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                init=init_method,
                tol=1e-6  # Tighter convergence tolerance
            )
            
            labels = kmeans.fit_predict(data_scaled)
            
            # Evaluate clustering quality
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(data_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_kmeans = kmeans
        
        if best_kmeans is None:
            # Fallback to basic k-means
            best_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            best_kmeans.fit(data_scaled)
        
        labels = best_kmeans.labels_
        return labels, data_scaled, best_score
        
    except Exception as e:
        raise Exception(f"Error in improved K-means clustering: {str(e)}")

def improved_spectral_clustering(data, n_clusters, random_state=42, **kwargs):
    """Optimized Spectral clustering for images"""
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        gamma = kwargs.get('gamma', 1.0)
        
        # Use built-in nearest neighbors mode
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=min(15, len(data_scaled)-1),
            gamma=gamma,
            random_state=random_state,
            assign_labels='kmeans',
            n_init=5
        )
        
        labels = clusterer.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else 0
        
        return labels, data_scaled, score
        
    except Exception as e:
        raise Exception(f"Efficient spectral clustering error: {str(e)}")
    
def dbscan_clustering(data, **kwargs):
    """DBSCAN for density-based clustering"""
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        
        score = silhouette_score(data_scaled, labels) if len(set(labels)) > 1 else 0
        return labels, data_scaled, score
        
    except Exception as e:
        raise Exception(f"Error in DBSCAN clustering: {str(e)}")

def louvain_clustering(data, **kwargs):
    """Louvain community detection algorithm for social media data"""
    try:
        # Normalize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Build similarity matrix
        threshold = kwargs.get('threshold', 0.7)
        resolution = kwargs.get('resolution', 1.0)
        
        # Calculate pairwise correlations
        correlation_matrix = np.corrcoef(data_scaled)
        
        # Create adjacency matrix based on threshold
        adjacency_matrix = np.abs(correlation_matrix) > threshold
        np.fill_diagonal(adjacency_matrix, False)  # Remove self-loops
        
        # Create networkx graph
        G = nx.Graph()
        n_nodes = len(data_scaled)
        G.add_nodes_from(range(n_nodes))
        
        # Add edges based on adjacency matrix
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adjacency_matrix[i, j]:
                    weight = abs(correlation_matrix[i, j])
                    G.add_edge(i, j, weight=weight)
        
        # Apply Louvain algorithm
        if len(G.edges()) > 0:
            communities = community_louvain.best_partition(G, resolution=resolution, random_state=42)
            labels = np.array([communities[i] for i in range(n_nodes)])
        else:
            # If no edges, assign each node to its own community
            labels = np.arange(n_nodes)
        
        # Calculate modularity score
        modularity = 0
        if len(G.edges()) > 0:
            modularity = community_louvain.modularity(communities, G)
        
        return labels, data_scaled, modularity, G
        
    except Exception as e:
        raise Exception(f"Error in Louvain clustering: {str(e)}")

def create_enhanced_graph_visualization(data, labels, mode, G=None, original_shape=None, original_image=None):
    """Create enhanced visualizations including network graphs"""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        
        if mode == 'image':
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original image
            if original_image is not None:
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
            
            # Clustered regions
            clustered_img = labels.reshape(original_shape[:2])
            im1 = axes[0, 1].imshow(clustered_img, cmap='viridis')
            axes[0, 1].set_title('Clustered Regions', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Color distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            axes[1, 0].pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
                          colors=colors, autopct='%1.1f%%')
            axes[1, 0].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
            
            # PCA visualization
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
            else:
                data_2d = data[:, :2]
                
            scatter = axes[1, 1].scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                                       cmap='viridis', alpha=0.7, s=20)
            axes[1, 1].set_title('Feature Space (PCA)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('First Principal Component')
            axes[1, 1].set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=axes[1, 1])
            
        else:  # Social media mode with network graph
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # PCA scatter plot
            if data.shape[1] > 2:
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(data)
            else:
                data_2d = data[:, :2]
            
            scatter = axes[0, 0].scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                                       cmap='tab10', alpha=0.7, s=50)
            axes[0, 0].set_title('Data Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('First Principal Component')
            axes[0, 0].set_ylabel('Second Principal Component')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Cluster distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            bars = axes[0, 1].bar(range(len(unique_labels)), counts, color=colors)
            axes[0, 1].set_title('Cluster Sizes', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Number of Points')
            axes[0, 1].set_xticks(range(len(unique_labels)))
            
            # Network graph visualization (if available)
            if G is not None and len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Draw nodes colored by community
                node_colors = [labels[node] for node in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     cmap=plt.cm.tab10, node_size=50, ax=axes[1, 0])
                
                # Draw edges with varying opacity based on weight
                edges = G.edges()
                if len(edges) > 0:
                    weights = [G[u][v].get('weight', 1) for u, v in edges]
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=weights, ax=axes[1, 0])
                
                axes[1, 0].set_title('Network Graph (Communities)', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
            else:
                axes[1, 0].text(0.5, 0.5, 'No network structure\ndetected', 
                               ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('Network Graph', fontsize=14, fontweight='bold')
            
            # Silhouette analysis
            try:
                if len(set(labels)) > 1:
                    sample_silhouette_values = silhouette_samples(data, labels)
                    
                    y_lower = 10
                    for i in range(len(unique_labels)):
                        cluster_silhouette_values = sample_silhouette_values[labels == unique_labels[i]]
                        cluster_silhouette_values.sort()
                        
                        size_cluster_i = cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i
                        
                        color = plt.cm.tab10(i / len(unique_labels))
                        axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper),
                                               0, cluster_silhouette_values,
                                               facecolor=color, edgecolor=color, alpha=0.7)
                        
                        axes[1, 1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(unique_labels[i]))
                        y_lower = y_upper + 10
                    
                    axes[1, 1].set_xlabel('Silhouette coefficient values')
                    axes[1, 1].set_ylabel('Cluster label')
                    axes[1, 1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Silhouette analysis\nnot available', 
                                   ha='center', va='center')
                    axes[1, 1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            except:
                axes[1, 1].text(0.5, 0.5, 'Silhouette analysis\nfailed', 
                               ha='center', va='center')
                axes[1, 1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        raise Exception(f"Error creating enhanced visualization: {str(e)}")

def extract_color_palette(image_array, n_colors=5):
    """Extract dominant colors from image using improved method"""
    try:
        pixels = image_array.reshape(-1, 3)
        
        # Use improved K-means for color extraction
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get color percentages
        labels = kmeans.labels_
        percentages = []
        for i in range(n_colors):
            percentage = (labels == i).sum() / len(labels) * 100
            percentages.append(round(percentage, 1))
        
        return colors.tolist(), percentages
    except Exception as e:
        return [], []

def process_social_media_data(file_path, file_type):
    """Enhanced social media data processing"""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_type == 'json':
            df = pd.read_json(file_path)
        else:
            raise Exception("Unsupported file type")
        
        # Clean the data
        df = df.dropna(thresh=len(df.columns) * 0.5)  # Drop rows with too many NaN values
        
        # Extract numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            # Create features from text columns
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                features = []
                feature_names = []
                
                for col in text_cols[:3]:  # Limit to first 3 text columns
                    if col in df.columns:
                        # Text length
                        text_len = df[col].astype(str).str.len()
                        features.append(text_len)
                        feature_names.append(f'{col}_length')
                        
                        # Word count
                        word_count = df[col].astype(str).str.split().str.len()
                        features.append(word_count)
                        feature_names.append(f'{col}_word_count')
                        
                        # Character diversity (unique chars / total chars)
                        char_diversity = df[col].astype(str).apply(
                            lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
                        )
                        features.append(char_diversity)
                        feature_names.append(f'{col}_char_diversity')
                
                if features:
                    feature_df = pd.concat(features, axis=1)
                    feature_df.columns = feature_names
                    return feature_df.fillna(0).values, df
        
        # Use numeric columns
        data = df[numeric_cols].fillna(df[numeric_cols].median()).values
        return data, df
        
    except Exception as e:
        raise Exception(f"Error processing social media data: {str(e)}")

def generate_mock_social_data(platform, data_type, search_query, num_points=200):
    """Generate enhanced mock social media data"""
    np.random.seed(42)
    data = []
    keywords = search_query.split() if search_query else ['social', 'media', 'analytics']
    
    for i in range(num_points):
        if data_type == 'posts':
            # Create correlated engagement metrics
            base_engagement = np.random.exponential(100)
            likes = int(base_engagement * np.random.uniform(0.6, 1.2))
            shares = int(base_engagement * np.random.uniform(0.1, 0.3))
            comments = int(base_engagement * np.random.uniform(0.05, 0.2))
            
            # Add time-based features
            hour_posted = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            
            data.append({
                'id': i,
                'text': f"Post about {np.random.choice(keywords)} #{np.random.choice(keywords)}",
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'engagement_score': likes + shares * 2 + comments * 3,
                'hour_posted': hour_posted,
                'day_of_week': day_of_week,
                'platform': platform,
                'sentiment_score': np.random.uniform(-1, 1),
                'text_length': np.random.randint(50, 280)
            })
            
        elif data_type == 'users':
            followers = int(np.random.exponential(1000))
            following = int(np.random.exponential(500))
            posts_count = int(np.random.exponential(100))
            
            # Calculate realistic engagement rate
            engagement_rate = np.random.beta(2, 20)  # Most users have low engagement
            
            data.append({
                'user_id': i,
                'followers': followers,
                'following': following,
                'posts_count': posts_count,
                'engagement_rate': engagement_rate,
                'follower_following_ratio': followers / (following + 1),
                'posts_per_day': posts_count / 365,
                'account_age_days': np.random.randint(30, 2000),
                'platform': platform,
                'verified': np.random.choice([0, 1], p=[0.95, 0.05])
            })
    
    return pd.DataFrame(data)

def analyze_louvain_communities(df, labels, G=None):
    """Analyze communities found by Louvain algorithm"""
    communities_info = []
    
    unique_labels = np.unique(labels)
    
    for community_id in unique_labels:
        community_mask = labels == community_id
        community_data = df[community_mask]
        community_size = len(community_data)
        
        # Extract keywords from text columns
        keywords = []
        text_cols = community_data.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            for text_col in text_cols[:2]:  # First 2 text columns
                if text_col in community_data.columns:
                    all_text = ' '.join(community_data[text_col].astype(str))
                    words = re.findall(r'\b\w+\b', all_text.lower())
                    word_counts = Counter(words)
                    keywords.extend([word for word, count in word_counts.most_common(3) 
                                   if len(word) > 3 and word not in ['post', 'about', 'sample']])
        
        # Analyze numeric characteristics
        characteristics = []
        numeric_cols = community_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:4]:  # Top 4 numeric features
            community_mean = community_data[col].mean()
            overall_mean = df[col].mean()
            
            ratio = community_mean / (overall_mean + 1e-8)
            
            if ratio > 1.5:
                characteristics.append(f"High {col.replace('_', ' ')}")
            elif ratio < 0.5:
                characteristics.append(f"Low {col.replace('_', ' ')}")
        
        # Network properties if graph is available
        network_props = {}
        if G is not None:
            community_nodes = [i for i, label in enumerate(labels) if label == community_id]
            if len(community_nodes) > 1:
                subgraph = G.subgraph(community_nodes)
                if len(subgraph.edges()) > 0:
                    network_props = {
                        'internal_edges': len(subgraph.edges()),
                        'density': nx.density(subgraph),
                        'avg_clustering': np.mean(list(nx.clustering(subgraph).values()))
                    }
        
        communities_info.append({
            'id': int(community_id),
            'size': community_size,
            'keywords': keywords[:5] if keywords else ['community', 'social', 'network'],
            'characteristics': characteristics[:3] if characteristics else ['Moderate activity'],
            'network_properties': network_props
        })
    
    return communities_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    start_time = time.time()
    
    try:
        mode = request.form.get('mode', 'image')
        algorithm = request.form.get('algorithm', 'kmeans')
        n_clusters = int(request.form.get('num_clusters', 3))
        random_state = int(request.form.get('random_state', 42))
        
        clustering_params = {}
        if algorithm == 'kmeans':
            clustering_params['max_iterations'] = int(request.form.get('max_iterations', 300))
            clustering_params['n_init'] = 20
        elif algorithm == 'spectral':
            clustering_params['gamma'] = float(request.form.get('gamma', 1.0))
            clustering_params['n_neighbors'] = int(request.form.get('n_neighbors', 10))
        elif algorithm == 'dbscan':
            clustering_params['eps'] = float(request.form.get('eps', 0.5))
            clustering_params['min_samples'] = int(request.form.get('min_samples', 5))
        
        if n_clusters < 2 or n_clusters > 15:
            return jsonify({'error': 'Number of clusters must be between 2 and 15'}), 400
        
        if mode == 'image':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file'}), 400
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Validate file content
            if not validate_file_content(file_path):
                os.remove(file_path)
                return jsonify({'error': 'Invalid file content'}), 400
            
            try:
                enhanced_data, original_shape, original_image, rgb_pixels = enhance_image_preprocessing(file_path)
                
                if algorithm == 'kmeans':
                    labels, data_scaled, quality_score = improved_kmeans_clustering(
                        enhanced_data, n_clusters, random_state, **clustering_params
                    )
                elif algorithm == 'spectral':
                    labels, data_scaled, quality_score = improved_spectral_clustering(
                        enhanced_data, n_clusters, random_state, **clustering_params
                    )
                elif algorithm == 'dbscan':
                    labels, data_scaled, quality_score = dbscan_clustering(
                        enhanced_data, **clustering_params
                    )
                    n_clusters = len(np.unique(labels))  # Actual clusters found by DBSCAN
                
                viz_base64 = create_enhanced_graph_visualization(
                    data_scaled, labels, 'image', None, original_shape, original_image
                )
                
                # Extract color palette with percentages
                color_palette, color_percentages = extract_color_palette(rgb_pixels, n_clusters)
                
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_stats = {f'Cluster {i}': int(count) for i, count in zip(unique_labels, counts)}
                processing_time = round(time.time() - start_time, 2)
                
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'mode': 'image',
                    'visualization': viz_base64,
                    'color_palette': color_palette,
                    'color_percentages': color_percentages,
                    'cluster_stats': cluster_stats,
                    'algorithm_used': algorithm.upper(),
                    'num_clusters': n_clusters,
                    'processing_time': processing_time,
                    'pixels_processed': len(enhanced_data),
                    'quality_score': round(quality_score, 3) if quality_score > 0 else 'N/A',
                    'download_available': False
                })
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
                
        else:  # Social media mode with Louvain algorithm
            platform = request.form.get('platform', 'twitter')
            data_type = request.form.get('data_type', 'posts')
            search_query = request.form.get('search_query', '')
            
            # Louvain-specific parameters
            threshold = float(request.form.get('threshold', 0.7))
            resolution = float(request.form.get('resolution', 1.0))
            
            if platform == 'upload':
                if 'file' not in request.files:
                    return jsonify({'error': 'No dataset file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '' or not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file'}), 400
                
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Validate file content
                if not validate_file_content(file_path):
                    os.remove(file_path)
                    return jsonify({'error': 'Invalid file content'}), 400
                
                try:
                    file_ext = filename.split('.')[-1].lower()
                    data, original_df = process_social_media_data(file_path, file_ext)
                    os.remove(file_path)
                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise e
            else:
                if not search_query.strip():
                    return jsonify({'error': 'Search query is required for demo data generation'}), 400
                
                original_df = generate_mock_social_data(platform, data_type, search_query)
                numeric_cols = original_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = original_df[numeric_cols].fillna(0).values
                else:
                    return jsonify({'error': 'No numeric features found for clustering'}), 400
            
            # Apply Louvain algorithm
            labels, data_scaled, modularity, G = louvain_clustering(
                data, threshold=threshold, resolution=resolution
            )

            viz_base64 = create_enhanced_graph_visualization(
                data_scaled, labels, 'social', G=G
            )

            communities_info = analyze_louvain_communities(original_df, labels, G=G)

            processing_time = round(time.time() - start_time, 2)
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_stats = {f'Community {i}': int(count) for i, count in zip(unique_labels, counts)}

            return jsonify({
                'success': True,
                'mode': 'social',
                'visualization': viz_base64,
                'communities': communities_info,
                'cluster_stats': cluster_stats,
                'algorithm_used': 'LOUVAIN',
                'modularity_score': round(modularity, 3) if modularity > 0 else 'N/A',
                'num_communities': len(unique_labels),
                'processing_time': processing_time,
                'points_processed': len(data_scaled),
                'download_available': False
            })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Processing failed',
        'details': str(e),
        'resolution': 'Try reducing cluster count or file size'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)