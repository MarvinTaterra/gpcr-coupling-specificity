import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns


def parse_residue_pair(pair_str):
    numbers = re.findall(r'Res (\d+)', pair_str)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None


def is_numeric_bw(val):
    if val is None:
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def normalize_bw(bw_val):
    if bw_val is None:
        return None
    if is_numeric_bw(bw_val):
        return f"{float(bw_val):.2f}"
    return str(bw_val).strip()


def load_single_system(folder_path, nmi_file, mapping_file):
    folder = Path(folder_path)
    nmi_df = pd.read_csv(folder / nmi_file)
    with open(folder / mapping_file, 'r') as f:
        genetic_dict = json.load(f)
    return nmi_df, genetic_dict


def map_to_bw_pairs(nmi_df, genetic_dict):
    bw_pairs = {}
    
    nmi_col = None
    for col in nmi_df.columns:
        if 'mi' in col.lower() or 'nmi' in col.lower():
            nmi_col = col
            break
    
    if nmi_col is None:
        return bw_pairs
    
    for _, row in nmi_df.iterrows():
        try:
            res1, res2 = parse_residue_pair(row['Residue Pair'])
            if res1 is None or res2 is None:
                continue
            mi = row[nmi_col]
            bw1 = genetic_dict.get(str(res1))
            bw2 = genetic_dict.get(str(res2))
            if bw1 is not None and bw2 is not None:
                bw1_norm = normalize_bw(bw1)
                bw2_norm = normalize_bw(bw2)
                if bw1_norm and bw2_norm:
                    key = tuple(sorted([bw1_norm, bw2_norm]))
                    bw_pairs[key] = mi
        except:
            continue
    return bw_pairs

def load_all_systems(config):
    systems_data = {}
    
    for coupling_class, receptors in config.items():
        print(f"\nLoading {coupling_class} receptors...")
        
        for folder_path, nmi_file, mapping_file in receptors:
            receptor_name = Path(folder_path).name
            
            try:
                nmi_df, genetic_dict = load_single_system(folder_path, nmi_file, mapping_file)
                bw_pairs = map_to_bw_pairs(nmi_df, genetic_dict)
                
                systems_data[receptor_name] = {
                    'bw_pairs': bw_pairs,
                    'coupling': coupling_class,
                    'n_pairs': len(bw_pairs)
                }
                print(f"  ✓ {receptor_name}: {len(bw_pairs)} BW pairs")
                
            except Exception as e:
                print(f"  ✗ {receptor_name}: {e}")
    
    return systems_data

def build_feature_matrix(systems_data, min_presence=0.5):
    pair_counts = defaultdict(int)
    for name, data in systems_data.items():
        for pair in data['bw_pairs'].keys():
            pair_counts[pair] += 1
    
    n_systems = len(systems_data)
    min_count = int(min_presence * n_systems)
    selected_pairs = sorted([p for p, c in pair_counts.items() if c >= min_count])
    
    print(f"Total unique BW pairs: {len(pair_counts)}")
    print(f"Pairs in >= {min_presence*100:.0f}% of receptors: {len(selected_pairs)}")
    
    X = np.full((n_systems, len(selected_pairs)), np.nan)
    y = []
    sample_names = []
    
    for i, (name, data) in enumerate(systems_data.items()):
        sample_names.append(name)
        y.append(data['coupling'])
        for j, pair in enumerate(selected_pairs):
            if pair in data['bw_pairs']:
                X[i, j] = data['bw_pairs'][pair]
    
    feature_names = [f"{p[0]}_{p[1]}" for p in selected_pairs]
    return X, np.array(y), feature_names, sample_names


def plot_unsupervised_clustering(
    X_scaled,
    y,
    sample_names,
    colors_true=None,
    n_clusters=3,
    save_path='unsupervised_clustering.png',
    random_state=42,
    figsize=(16, 12)
):
    """
    Create PCA, t-SNE, K-Means, and hierarchical clustering figures.

    Parameters
    ----------
    X_scaled : array-like, shape (n_samples, n_features)
        Scaled feature matrix.
    y : array-like, shape (n_samples,)
        True class labels.
    sample_names : list of str
        Sample identifiers for dendrogram labeling.
    colors_true : dict, optional
        Mapping of class labels to colors.
    n_clusters : int, default=3
        Number of clusters for K-Means.
    save_path : str, default='unsupervised_clustering.png'
        Path to save the figure.
    random_state : int, default=42
        Random seed for reproducibility.
    figsize : tuple, default=(16, 12)
        Figure size.
    """

    if colors_true is None:
        colors_true = {
            'Gs': '#3498db',
            'Gi': '#2ecc71',
            'Gq': '#e74c3c'
        }

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    fig = plt.figure(figsize=figsize)

    # 1. PCA
    ax1 = fig.add_subplot(2, 2, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    for cls in le.classes_:
        mask = y == cls
        ax1.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors_true.get(cls, '#7f8c8d'),
            label=cls,
            alpha=0.7,
            s=60,
            edgecolors='white'
        )

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('PCA - Colored by True Labels')
    ax1.legend()

    # 2. t-SNE
    ax2 = fig.add_subplot(2, 2, 2)
    tsne = TSNE(
        n_components=2,
        perplexity=min(15, len(X_scaled) - 1),
        random_state=random_state,
        max_iter=1000
    )
    X_tsne = tsne.fit_transform(X_scaled)

    for cls in le.classes_:
        mask = y == cls
        ax2.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=colors_true.get(cls, '#7f8c8d'),
            label=cls,
            alpha=0.7,
            s=60,
            edgecolors='white'
        )

    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE - Colored by True Labels')
    ax2.legend()

    # 3. K-Means (shown in PCA space)
    ax3 = fig.add_subplot(2, 2, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    colors_km = ['#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    for i in range(n_clusters):
        mask = kmeans_labels == i
        ax3.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors_km[i % len(colors_km)],
            label=f'Cluster {i}',
            alpha=0.7,
            s=60,
            edgecolors='white'
        )

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax3.set_title(f'K-Means (k={n_clusters}) - PCA Space')
    ax3.legend()

    # 4. Dendrogram
    ax4 = fig.add_subplot(2, 2, 4)
    labels_dendro = [
        f"{name.split('_')[0]}({y[i]})"
        for i, name in enumerate(sample_names)
    ]
    linkage_matrix = linkage(X_scaled, method='ward')
    dendrogram(
        linkage_matrix,
        labels=labels_dendro,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax4
    )
    ax4.set_title('Hierarchical Clustering (Ward)')
    ax4.set_ylabel('Distance')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return {
        "label_encoder": le,
        "y_enc": y_enc,
        "kmeans_labels": kmeans_labels,
        "pca": pca,
        "tsne": tsne
    }


def plot_optimal_number_of_clusters(
    X_scaled,
    k_range=range(2, 8),
    true_k=3,
    save_path="optimal_clusters.png",
    random_state=42
):
    """
    Plot silhouette scores and inertia to estimate optimal number of clusters.
    """

    silhouettes = []
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, labels))
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Silhouette plot
    axes[0].plot(k_range, silhouettes, 'bo-', markersize=8)
    axes[0].axvline(true_k, color='red', linestyle='--', label=f'k={true_k}')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs k')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Elbow plot
    axes[1].plot(k_range, inertias, 'go-', markersize=8)
    axes[1].axvline(true_k, color='red', linestyle='--', label=f'k={true_k}')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Inertia')
    axes[1].set_title('Elbow Method')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    best_k = list(k_range)[np.argmax(silhouettes)]

    return {
        "k_range": list(k_range),
        "silhouettes": silhouettes,
        "inertias": inertias,
        "best_k": best_k
    }


def plot_pca_with_sample_labels(
    X_pca,
    y,
    sample_names,
    pca,
    colors_true,
    save_path="pca_with_labels.png",
    figsize=(14, 10)
):
    """
    Plot a PCA scatter plot colored by true labels with sample annotations.

    Parameters
    ----------
    X_pca : array-like, shape (n_samples, 2)
        PCA-transformed data.
    y : array-like, shape (n_samples,)
        True class labels.
    sample_names : list of str
        Sample identifiers.
    pca : fitted sklearn PCA object
        Used for explained variance in axis labels.
    colors_true : dict
        Mapping of class labels to colors.
    save_path : str
        File path for saving the figure.
    figsize : tuple
        Figure size.
    """

    fig, ax = plt.subplots(figsize=figsize)

    for cls in set(y):
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors_true.get(cls, '#7f8c8d'),
            label=cls,
            alpha=0.7,
            s=100,
            edgecolors='white'
        )

    # Add sample (receptor) names
    for i, name in enumerate(sample_names):
        short_name = name.split('_')[0]
        ax.annotate(
            short_name,
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=7,
            alpha=0.8,
            ha='center',
            va='bottom'
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA of GPCR BW features - Colored by Gα-Coupling')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



RECEPTOR_CONFIG = {
    # Gs-coupled receptors (23)
    'Gs': [
        ("./Gs/5HT7R_7XTC_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/AA2AR_5G53_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/ADRB3_9IJE_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/CNR1_8K8J_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/EDNRA_8XVI_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/EDNRB_8XVH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/FFAR4_8H4I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GP119_7WCM_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GP139_7VUH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GP174_8KH5_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GPBAR_9GYO_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GPR3_8WW2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GPR6_8TYW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GPR12_7Y3G_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/GPR52_8HMP_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/NK1R_8U26_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/PE2R2_7CX2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/PE2R4_8GDB_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/RXFP1_7TMW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/SUCR1_8JPP_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/TAAR1_8W89_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gs/V2R_7DW9_DONE", "nmi_df.csv", "genetic_mapping.json")
    ],
    
    'Gi': [
        ("./Gi/AA1R_7LD4_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/AA3R_8X17_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/ADA2A_9CBL_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/CCR6_6WWZ_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/CNR1_9ERX_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/EDNRB_8IY5_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/FFAR1_9K1C_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/FFAR2_9K1D_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/MCHR1_8WWK_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/OPRK_8F7W_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/OPRM_8F7Q_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/PE2R4_8GCP_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/PTAFR_8XYD_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/5HT4R_7XTA_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/NMUR2_7XK8_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/NTR1_6OS9_PREAPRED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/P2Y12_7XXI_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gi/PE2R3_8GDC_PREPARED", "nmi_df.csv", "genetic_mapping.json")
    ],
    
    'Gq': [
        ("./Gq/5HT2A_8UWL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/5HT2B_7SRR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/ADA1A_8THL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/BKRB2_7F6I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/EDNRA_8HCQ_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/EDNRB_8HCX_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/FFAR4_8G59_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/HRH2_8YN4_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/KISSR_8XGS_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/MCHR2_8WST_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/MRGX1_8JGF_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/MRGX2_7S8L_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/MTLR_8IBV_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/NMUR1_7W56_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/NMUR2_7W55_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/P2RY1_7XXH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/PAR1_8XOR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/PF2R_8IUK_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/PRLHR_8ZPT_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("./Gq/SSR2_7Y27_DONE", "nmi_df.csv", "genetic_mapping.json")
    ]
}

systems_data = load_all_systems(RECEPTOR_CONFIG)
X, y, feature_names, sample_names = build_feature_matrix(systems_data, min_presence=0.5)
print(f"\nFeature matrix: {X.shape}")
print(f"Missing values: {np.isnan(X).sum()} ({np.isnan(X).mean()*100:.1f}%)")
# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Remove constant features
variances = np.var(X_imputed, axis=0)
keep_mask = variances > 1e-10
X_filtered = X_imputed[:, keep_mask]
filtered_feature_names = [f for f, k in zip(feature_names, keep_mask) if k]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

print(f"Final feature matrix: {X_scaled.shape}")

results = plot_unsupervised_clustering(
    X_scaled,
    y,
    sample_names
)

y_enc = results["y_enc"]
kmeans_labels = results["kmeans_labels"]
print("=" * 60)
print("CLUSTERING QUALITY METRICS")
print("=" * 60)

ari_km = adjusted_rand_score(y_enc, kmeans_labels)
nmi_km = normalized_mutual_info_score(y_enc, kmeans_labels)
print(f"\nK-Means (k=3):")
print(f"  Adjusted Rand Index: {ari_km:.3f}  (1.0 = perfect, 0 = random)")
print(f"  Normalized Mutual Info: {nmi_km:.3f}  (1.0 = perfect, 0 = no info)")

hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)

ari_hier = adjusted_rand_score(y_enc, hier_labels)
nmi_hier = normalized_mutual_info_score(y_enc, hier_labels)
print(f"\nHierarchical (Ward, k=3):")
print(f"  Adjusted Rand Index: {ari_hier:.3f}")
print(f"  Normalized Mutual Info: {nmi_hier:.3f}")

print("=" * 60)
print("OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

opt_results = plot_optimal_number_of_clusters(
    X_scaled,
    k_range=range(2, 8),
    true_k=3
)

print(f"\nBest k by silhouette score: {opt_results['best_k']}")
print(f"Silhouette at k=3: {opt_results['silhouettes'][1]:.3f}")

results = plot_unsupervised_clustering(X_scaled, y, sample_names)

plot_pca_with_sample_labels(
    X_pca=results["pca"].transform(X_scaled),
    y=y,
    sample_names=sample_names,
    pca=results["pca"],
    colors_true={
        'Gs': '#3498db',
        'Gi': '#2ecc71',
        'Gq': '#e74c3c'
    }
)
