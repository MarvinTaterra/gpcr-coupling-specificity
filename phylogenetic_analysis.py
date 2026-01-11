import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage
from Bio import Phylo, AlignIO, SeqIO, pairwise2
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, DistanceMatrix
from Bio import Entrez
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def fetch_uniprot_sequence(uniprot_id, retries=3):
    """
    Fetch protein sequence from UniProt REST API.
    Returns sequence string or None if failed.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Parse FASTA format
                lines = response.text.strip().split('\n')
                sequence = ''.join(lines[1:])  # Skip header line
                return sequence
            elif response.status_code == 404:
                print(f"  ✗ {uniprot_id}: Not found in UniProt")
                return None
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1} for {uniprot_id}...")
                continue
            else:
                print(f"  ✗ {uniprot_id}: Failed after {retries} attempts - {e}")
                return None
    
    return None

def calculate_sequence_identity(seq1, seq2):
    """
    Calculate sequence identity between two sequences using pairwise alignment.
    Returns identity score (0-1).
    """
    # Use BLOSUM62 matrix for protein alignment
    alignments = pairwise2.align.globalxx(seq1, seq2)
    
    if not alignments:
        return 0.0
    
    # Get best alignment
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
    
    # Calculate identity
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-')
    total = len([a for a in aligned_seq1 if a != '-'])
    
    if total == 0:
        return 0.0
    
    return matches / total

def calculate_patristic_distance(tree, name1, name2):
    """
    Calculate patristic distance (sum of branch lengths) between two taxa.
    """
    # Find the two terminal nodes
    node1 = None
    node2 = None
    
    for terminal in tree.get_terminals():
        if terminal.name == name1:
            node1 = terminal
        if terminal.name == name2:
            node2 = terminal
    
    if node1 is None or node2 is None:
        return None
    
    # Calculate distance
    return tree.distance(node1, node2)

def parse_residue_pair(pair_str):
    numbers = re.findall(r'Res (\d+)', pair_str)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def normalize_bw(bw_val):
    if bw_val is None:
        return None
    try:
        return f"{float(bw_val):.2f}"
    except:
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


UNIPROT_IDS = {
    # Aminergic receptors
    '5HT2A': 'P28223',
    '5HT2B': 'P41595',
    '5HT4R': 'Q13639',
    '5HT7R': 'P34969',
    'ADA1A': 'P35348',
    'ADA2A': 'P08913',
    'ADRB3': 'P13945',
    'HRH2': 'P25021',
    
    # Adenosine receptors
    'AA1R': 'P30542',
    'AA2AR': 'P29274',
    'AA3R': 'P0DMS8',
    
    # Opioid receptors
    'OPRK': 'P41145',
    'OPRM': 'P35372',
    
    # Endothelin receptors
    'EDNRA': 'P25101',
    'EDNRB': 'P24530',
    
    # Cannabinoid
    'CNR1': 'P21554',
    
    # Free fatty acid receptors
    'FFAR1': 'O14842',
    'FFAR2': 'O15552',
    'FFAR4': 'Q5NUL3',
    
    # Melanocortin
    'MCHR1': 'Q99705',
    'MCHR2': 'Q969V1',
    
    # Neuromedin U
    'NMUR1': 'Q9HB89',
    'NMUR2': 'Q9GZQ4',
    
    # Prostaglandin receptors
    'PE2R2': 'P43116',
    'PE2R3': 'P43115',
    'PE2R4': 'P35408',
    'PF2R': 'P43088',
    
    # Purinergic
    'P2Y12': 'Q9H244',
    'P2RY1': 'P47900',
    
    # Orphan GPCRs
    'GP119': 'Q8TDV5',
    'GP139': 'Q6DWJ6',
    'GP174': 'Q9BXC1',
    'GPR3': 'P46089',
    'GPR6': 'P46093',
    'GPR12': 'P47775',
    'GPR52': 'Q9Y2T5',
    'GPBAR': 'Q8TDU5',
    
    # Other receptors
    'BKRB2': 'P30411',
    'CCR6': 'P51684',
    'KISSR': 'Q969F8',
    'MRGX1': 'Q96LB1',
    'MRGX2': 'Q96LB0',
    'MTLR': 'P48039',
    'NK1R': 'P25103',
    'NTR1': 'P30989',
    'PAR1': 'P25116',
    'PRLHR': 'Q8TCW9',
    'PTAFR': 'P25105',
    'RXFP1': 'Q9HBX9',
    'SSR2': 'P30874',
    'SUCR1': 'Q8TE23',
    'TAAR1': 'Q96RJ0',
    'V2R': 'P30518',
}

RECEPTOR_CONFIG = {
    'Gs': [
       # ("C:/Users/Marvin/Desktop/Compute folder/Gs/5HT4R_7XT8_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/5HT7R_7XTC_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/AA2AR_5G53_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/ADRB3_9IJE_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/CNR1_8K8J_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/EDNRA_8XVI_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/EDNRB_8XVH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/FFAR4_8H4I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GP119_7WCM_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GP139_7VUH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GP174_8KH5_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GPBAR_9GYO_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GPR3_8WW2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GPR6_8TYW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GPR12_7Y3G_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/GPR52_8HMP_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/NK1R_8U26_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/PE2R2_7CX2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/PE2R4_8GDB_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/RXFP1_7TMW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/SUCR1_8JPP_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/TAAR1_8W89_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gs/V2R_7DW9_DONE", "nmi_df.csv", "genetic_mapping.json")
    ],
    'Gi': [
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/AA1R_7LD4_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/AA3R_8X17_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/ADA2A_9CBL_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/CCR6_6WWZ_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/CNR1_9ERX_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/EDNRB_8IY5_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/FFAR1_9K1C_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/FFAR2_9K1D_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/MCHR1_8WWK_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/OPRK_8F7W_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/OPRM_8F7Q_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/PE2R4_8GCP_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/PTAFR_8XYD_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/5HT4R_7XTA_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/NMUR2_7XK8_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/NTR1_6OS9_PREAPRED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/P2Y12_7XXI_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gi/PE2R3_8GDC_PREPARED", "nmi_df.csv", "genetic_mapping.json")
    ],
    'Gq': [
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/5HT2A_8UWL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/5HT2B_7SRR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/ADA1A_8THL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/BKRB2_7F6I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/EDNRA_8HCQ_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/EDNRB_8HCX_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/FFAR4_8G59_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/HRH2_8YN4_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/KISSR_8XGS_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/MCHR2_8WST_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/MRGX1_8JGF_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/MRGX2_7S8L_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/MTLR_8IBV_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/NMUR1_7W56_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/NMUR2_7W55_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/P2RY1_7XXH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/PAR1_8XOR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/PF2R_8IUK_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/PRLHR_8ZPT_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("C:/Users/Marvin/Desktop/Compute folder/Gq/SSR2_7Y27_DONE", "nmi_df.csv", "genetic_mapping.json")
    ]
}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

sequences = {}
failed = []

for receptor, uniprot_id in UNIPROT_IDS.items():
    sequence = fetch_uniprot_sequence(uniprot_id)
    if sequence:
        sequences[receptor] = sequence
        print(f"  ✓ {receptor} ({uniprot_id}): {len(sequence)} aa")
    else:
        failed.append(receptor)

print(f"\n" + "="*60)
print(f"Successfully fetched: {len(sequences)}/{len(UNIPROT_IDS)} sequences")

# Save sequences to file
with open('gpcr_sequences.fasta', 'w') as f:
    for receptor, seq in sequences.items():
        f.write(f">{receptor}\n{seq}\n")
print("\nSequences saved to: gpcr_sequences.fasta")


# Calculate pairwise sequence identities
receptor_list = sorted(sequences.keys())
n_receptors = len(receptor_list)

# Initialize distance matrix
sequence_distance_matrix = np.zeros((n_receptors, n_receptors))

# Calculate pairwise distances
total_pairs = n_receptors * (n_receptors - 1) // 2
completed = 0

for i in range(n_receptors):
    for j in range(i+1, n_receptors):
        rec1, rec2 = receptor_list[i], receptor_list[j]
        
        # Calculate sequence identity
        identity = calculate_sequence_identity(sequences[rec1], sequences[rec2])
        
        # Convert identity to distance (0 = identical, 1 = completely different)
        distance = 1.0 - identity
        
        # Fill symmetric matrix
        sequence_distance_matrix[i, j] = distance
        sequence_distance_matrix[j, i] = distance
        
        completed += 1
        if completed % 100 == 0:
            print(f"  Progress: {completed}/{total_pairs} pairs ({completed/total_pairs*100:.1f}%)")


# Convert numpy matrix to BioPython DistanceMatrix format
matrix_data = []
for i in range(n_receptors):
    row = [sequence_distance_matrix[i, j] for j in range(i+1)]
    matrix_data.append(row)

dm = DistanceMatrix(receptor_list, matrix_data)

# Build tree using Neighbor-Joining
constructor = DistanceTreeConstructor()
nj_tree = constructor.nj(dm)

# Save tree to file
Phylo.write(nj_tree, 'gpcr_phylogenetic_tree.xml', 'phyloxml')
print("Tree saved to: gpcr_phylogenetic_tree.xml")

patristic_distance_matrix = np.zeros((n_receptors, n_receptors))

for i in range(n_receptors):
    for j in range(i+1, n_receptors):
        rec1, rec2 = receptor_list[i], receptor_list[j]
        distance = calculate_patristic_distance(nj_tree, rec1, rec2)
        
        if distance is not None:
            patristic_distance_matrix[i, j] = distance
            patristic_distance_matrix[j, i] = distance

patristic_df = pd.DataFrame(patristic_distance_matrix, 
                            index=receptor_list, 
                            columns=receptor_list)
patristic_df.to_csv('patristic_distances.csv')
print("\nPatristic distances saved to: patristic_distances.csv")#


systems_data = {}

for coupling, receptors in RECEPTOR_CONFIG.items():
    for folder_path, nmi_file, mapping_file in receptors:
        receptor_name = Path(folder_path).name.split('_')[0]
        try:
            nmi_df, genetic_dict = load_single_system(folder_path, nmi_file, mapping_file)
            bw_pairs = map_to_bw_pairs(nmi_df, genetic_dict)
            systems_data[Path(folder_path).name] = {
                'receptor': receptor_name,
                'coupling': coupling,
                'bw_pairs': bw_pairs
            }
        except Exception as e:
            print(f"  ✗ {receptor_name}: {e}")

# Build feature matrix
all_bw_keys = set()
for data in systems_data.values():
    all_bw_keys.update(data['bw_pairs'].keys())

pair_counts = defaultdict(int)
for data in systems_data.values():
    for key in data['bw_pairs'].keys():
        pair_counts[key] += 1

min_count = int(0.5 * len(systems_data))
selected_pairs = sorted([k for k, c in pair_counts.items() if c >= min_count])

# Build matrix
X = np.full((len(systems_data), len(selected_pairs)), np.nan)
sample_names = []
receptor_names = []
coupling_labels = []

for i, (sys_name, data) in enumerate(systems_data.items()):
    sample_names.append(sys_name)
    receptor_names.append(data['receptor'])
    coupling_labels.append(data['coupling'])
    for j, pair in enumerate(selected_pairs):
        if pair in data['bw_pairs']:
            X[i, j] = data['bw_pairs'][pair]

# Impute and scale
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"\nMI Feature matrix: {X_scaled.shape}")
print(f"Samples: {len(sample_names)}")
print(f"Unique receptors: {len(set(receptor_names))}")
# Calculate pairwise Euclidean distances in MI feature space
mi_dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

# Find common receptors
phylo_receptors = set(receptor_list)  # From sequences
mi_receptors = set(receptor_names)  # From MI data
common_receptors = sorted(phylo_receptors & mi_receptors)

# Create aligned matrices for common receptors
# Map receptor to index in each matrix
phylo_receptor_to_idx = {rec: i for i, rec in enumerate(receptor_list)}
mi_receptor_to_idx = {rec: i for i, rec in enumerate(receptor_names)}

common_phylo_indices = []
common_mi_indices = []

for rec in common_receptors:
    phylo_idx = phylo_receptor_to_idx[rec]
    # Find first occurrence in MI data
    mi_idx = receptor_names.index(rec)
    common_phylo_indices.append(phylo_idx)
    common_mi_indices.append(mi_idx)

# Extract aligned submatrices
phylo_dist_aligned = patristic_distance_matrix[np.ix_(common_phylo_indices, common_phylo_indices)]
mi_dist_aligned = mi_dist_matrix[np.ix_(common_mi_indices, common_mi_indices)]

print(f"\nAligned matrices shape: {phylo_dist_aligned.shape}")
print(f"Ready for correlation analysis on {len(common_receptors)} receptors")

# Extract upper triangular values (exclude diagonal)
n_common = len(common_receptors)
triu_indices = np.triu_indices(n_common, k=1)
phylo_dists = phylo_dist_aligned[triu_indices]
mi_dists = mi_dist_aligned[triu_indices]

# Calculate correlations
spearman_r, spearman_p = spearmanr(phylo_dists, mi_dists)
pearson_r, pearson_p = pearsonr(phylo_dists, mi_dists)

# Calculate effect size
r_squared = spearman_r ** 2

print(f"\nSpearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.2e}")
print(f"Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.2e}")
print(f"\nEffect size (R²): {r_squared:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Dendrogram - Phylogenetic
phylo_linkage = linkage(squareform(phylo_dist_aligned), method='average')
dendrogram(phylo_linkage, labels=common_receptors, leaf_rotation=90, 
          leaf_font_size=6, ax=ax1, color_threshold=0)
ax1.set_title('Phylogenetic Clustering\n(Sequence-based)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Patristic Distance')

# 2. Dendrogram - MDPATH
mi_linkage = linkage(squareform(mi_dist_aligned), method='ward')
dendrogram(mi_linkage, labels=common_receptors, leaf_rotation=90, 
          leaf_font_size=6, ax=ax2, color_threshold=0)
ax2.set_title('MDPath based clustering', fontsize=12, fontweight='bold')
ax2.set_ylabel('MI Distance')

plt.tight_layout()
plt.savefig('sequence_phylogeny_pathway_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved: sequence_phylogeny_pathway_correlation.png")

known_families = {
    'Opioid receptor': ['OPRK', 'OPRM'],
    'Endothelin receptor': ['EDNRA', 'EDNRB'],
    'Adenosine receptor': ['AA1R', 'AA2AR', 'AA3R'],
    'Serotonin receptor': ['5HT2A', '5HT2B', '5HT4R', '5HT7R'],
    'Neuromedin U receptor': ['NMUR1', 'NMUR2'],
    'Free fatty acid receptor': ['FFAR1', 'FFAR2', 'FFAR4'],
    'Melanin-concentrating hormone receptor': ['MCHR1', 'MCHR2'],
    'Prostaglandin receptor': ['PE2R2', 'PE2R3', 'PE2R4', 'PF2R'],
}

print("="*70)
print("WITHIN-FAMILY ANALYSIS: KNOWN PHYLOGENETIC GROUPS")
print("="*70)

family_results = []

for family_name, members in known_families.items():
    # Find members in common receptors
    present_members = [m for m in members if m in common_receptors]
    
    if len(present_members) < 2:
        continue
    
    # Get indices
    member_indices = [common_receptors.index(m) for m in present_members]
    
    # Within-family distances
    within_phylo = []
    within_mi = []
    for i in range(len(member_indices)):
        for j in range(i+1, len(member_indices)):
            idx_i, idx_j = member_indices[i], member_indices[j]
            within_phylo.append(phylo_dist_aligned[idx_i, idx_j])
            within_mi.append(mi_dist_aligned[idx_i, idx_j])
    
    # Between-family distances
    other_indices = [i for i in range(n_common) if i not in member_indices]
    between_phylo = []
    between_mi = []
    for i in member_indices:
        for j in other_indices:
            between_phylo.append(phylo_dist_aligned[i, j])
            between_mi.append(mi_dist_aligned[i, j])
    
    within_phylo_mean = np.mean(within_phylo)
    within_mi_mean = np.mean(within_mi)
    between_mi_mean = np.mean(between_mi)
    
    ratio = within_mi_mean / between_mi_mean
    
    # Statistical test
    if len(within_mi) > 0 and len(between_mi) > 0:
        u_stat, u_p = mannwhitneyu(within_mi, between_mi, alternative='less')
    else:
        u_stat, u_p = np.nan, np.nan
    
    family_results.append({
        'Family': family_name,
        'Members': ', '.join(present_members),
        'N': len(present_members),
        'Phylo_dist': within_phylo_mean,
        'MI_within': within_mi_mean,
        'MI_between': between_mi_mean,
        'Ratio': ratio,
        'P_value': u_p
    })
    
    print(f"\n{family_name} ({len(present_members)} members): {', '.join(present_members)}")
    print(f"  Evolutionary distance: {within_phylo_mean:.3f}")
    print(f"  MI distance (within): {within_mi_mean:.3f}")
    print(f"  MI distance (to others): {between_mi_mean:.3f}")
    print(f"  Ratio: {ratio:.3f} (< 1.0 = family clusters together)")
    print(f"  P-value: {u_p:.2e}")
    if u_p < 0.05:
        print(f"  ✓ SIGNIFICANT clustering by family")

# Save results
family_df = pd.DataFrame(family_results).sort_values('Ratio')
family_df.to_csv('phylogenetic_family_clustering_results.csv', index=False)
print("\n\nDetailed results saved: phylogenetic_family_clustering_results.csv")


# A
fig = plt.figure(figsize=(10, 4))


ax1 = plt.subplot(1, 2, 1)


xy = np.vstack([phylo_dists, mi_dists])
density = gaussian_kde(xy)(xy)


idx = density.argsort()
x_sorted = phylo_dists[idx]
y_sorted = mi_dists[idx]
density_sorted = density[idx]

sc = ax1.scatter(x_sorted, y_sorted, c=density_sorted, s=10, cmap='viridis',
                 alpha=0.8, edgecolors='none', rasterized=True)

xi, yi = np.mgrid[phylo_dists.min():phylo_dists.max():100j,
                  mi_dists.min():mi_dists.max():100j]
zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
contours = ax1.contour(xi, yi, zi, levels=5, colors='white', 
                       linewidths=0.6, alpha=0.7)

cb = plt.colorbar(sc, ax=ax1, pad=0.02, aspect=15)
cb.set_label('Density', fontsize=9)
cb.ax.tick_params(labelsize=8)


z = np.polyfit(phylo_dists, mi_dists, 1)
p = np.poly1d(z)
x_line = np.linspace(phylo_dists.min(), phylo_dists.max(), 100)
ax1.plot(x_line, p(x_line), 'r-', linewidth=1.2, alpha=0.9)


r_squared = spearman_r**2
if spearman_p < 0.001:
    text_str = f'ρ = {spearman_r:.3f}\nR² = {r_squared:.3f}\np < 0.001'
else:
    text_str = f'ρ = {spearman_r:.3f}\nR² = {r_squared:.3f}\np = {spearman_p:.3f}'

ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                   edgecolor='gray', linewidth=0.5))


ax1.set_xlabel('Phylogenetic Distance', fontsize=10, fontweight='bold')
ax1.set_ylabel('Pathway Distance', fontsize=10, fontweight='bold')
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left', pad=10)
ax1.tick_params(labelsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

#  B
ax2 = plt.subplot(1, 2, 2)
family_df_sorted = pd.DataFrame(family_results).sort_values('Ratio')

ax2.axvspan(0, 1.0, alpha=0.15, color='#2E7D32', zorder=0)

colors = ['#2E7D32' if p < 0.05 else '#909090' 
          for p in family_df_sorted['P_value']]

y_pos = np.arange(len(family_df_sorted))
bars = ax2.barh(y_pos, family_df_sorted['Ratio'], 
                color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

for i, (bar, ratio) in enumerate(zip(bars, family_df_sorted['Ratio'])):
    width = bar.get_width()
    ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
            f'{ratio:.2f}', va='center', ha='left', fontsize=8)

ax2.axvline(x=1.0, color='#D32F2F', linestyle='--', linewidth=1.2, alpha=0.9)

ax2.set_yticks(y_pos)
family_labels = [f"{f} (n={n})" for f, n in 
                 zip(family_df_sorted['Family'], family_df_sorted['N'])]
ax2.set_yticklabels(family_labels, fontsize=9)

ax2.set_xlabel('Within/Between Family Distance Ratio', fontsize=10, fontweight='bold')
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left', pad=10)
ax2.tick_params(labelsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.set_xlim(0, max(family_df_sorted['Ratio']) * 1.18)

sig_patch = mpatches.Patch(color='#2E7D32', label='p < 0.05')
nonsig_patch = mpatches.Patch(color='#757575', label='p ≥ 0.05')
ax2.legend(handles=[sig_patch, nonsig_patch], 
           loc='upper right',
           fontsize=8, frameon=True, edgecolor='gray', fancybox=False,
           ncol=1)

plt.tight_layout()

# Save figure
plt.savefig('figure2_phylogenetic_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_phylogenetic_analysis.pdf', bbox_inches='tight')
plt.show()