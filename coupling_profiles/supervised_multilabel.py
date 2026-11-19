import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_score,
    cross_val_predict, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, balanced_accuracy_score, make_scorer, multilabel_confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

LABELS = ['Gi', 'Gq', 'Gs']
COUPLING = pd.read_csv('coupling_labels.csv').set_index('system')


def macro_balanced_accuracy(Y, P):
    """Balanced accuracy of each label, averaged over the labels (labels with one class in Y are skipped)."""
    Y, P = np.asarray(Y), np.asarray(P)
    return np.mean([balanced_accuracy_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if len(np.unique(Y[:, j])) == 2])


macro_bacc_scorer = make_scorer(macro_balanced_accuracy)


class LabelSetStratifiedKFold(StratifiedKFold):
    """StratifiedKFold on the label combination of a multilabel target."""
    def split(self, X, y, groups=None):
        y = np.asarray(y)
        return super().split(X, y if y.ndim == 1 else [''.join(map(str, r)) for r in y], groups)


def f_multilabel(X, Y):
    """ANOVA F-score averaged over the labels (p value: smallest over the labels)."""
    F, p = zip(*(f_classif(X, Y[:, j]) for j in range(Y.shape[1])))
    return np.mean(F, axis=0), np.min(p, axis=0)

def parse_residue_pair(pair_str):
    """Parse \"('Res 4', 'Res 20')\" -> (4, 20)"""
    numbers = re.findall(r'Res (\d+)', pair_str)
    return int(numbers[0]), int(numbers[1])


def is_numeric_bw(val):
    """Check if BW value is numeric (TM helix) vs string (loop/terminus)"""
    if val is None:
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def normalize_bw(bw_val):
    """Normalize BW value to consistent string format."""
    if bw_val is None:
        return None
    if is_numeric_bw(bw_val):
        return f"{float(bw_val):.2f}"
    return str(bw_val).strip()


def load_single_system(nmi_path, mapping_path):
    """Load NMI data and genetic mapping for a single receptor."""
    nmi_df = pd.read_csv(nmi_path)

    with open(mapping_path, 'r') as f:
        genetic_dict = json.load(f)
    genetic_dict = {int(k): v for k, v in genetic_dict.items()}

    parsed = nmi_df['Residue Pair'].apply(parse_residue_pair)
    nmi_df['res1'] = parsed.apply(lambda x: x[0])
    nmi_df['res2'] = parsed.apply(lambda x: x[1])

    return nmi_df, genetic_dict


def map_to_bw_pairs(nmi_df, genetic_dict):
    """Map residue pairs to BW number pairs with NMI values."""
    bw_nmi = {}

    for _, row in nmi_df.iterrows():
        bw1 = normalize_bw(genetic_dict.get(row['res1']))
        bw2 = normalize_bw(genetic_dict.get(row['res2']))

        if bw1 is None or bw2 is None:
            continue

        bw_pair = tuple(sorted([bw1, bw2]))

        nmi_col = 'MI Difference' if 'MI Difference' in nmi_df.columns else 'nmi'
        bw_nmi[bw_pair] = row[nmi_col]

    return bw_nmi

def load_all_systems(config):
    """Load all receptor systems from the configuration."""
    systems_data = {}
    failed = []

    for coupling_class, receptors in config.items():
        print(f"\nLoading {coupling_class} receptors...")

        for folder_path, nmi_file, mapping_file in receptors:
            receptor_name = Path(folder_path).name

            nmi_path = Path(folder_path) / nmi_file
            map_path = Path(folder_path) / mapping_file

            try:
                nmi_df, genetic_dict = load_single_system(nmi_path, map_path)
                systems_data[receptor_name] = (nmi_df, genetic_dict, coupling_class)
                print(f"  ✓ {receptor_name}: {len(nmi_df)} pairs, {len(genetic_dict)} mapped residues")
            except FileNotFoundError:
                print(f"  ✗ {receptor_name}: File not found")
                failed.append(receptor_name)
            except Exception as e:
                print(f"  ✗ {receptor_name}: Error - {e}")
                failed.append(receptor_name)

    print(f"\n{'='*50}")
    print(f"Successfully loaded: {len(systems_data)} receptors")
    if failed:
        print(f"Failed to load: {len(failed)} receptors: {failed}")

    return systems_data

def build_feature_matrix(systems_data, min_presence=0.5):
    """Build the feature matrix of BW pairs present in at least min_presence of the systems."""
    all_bw_nmi = {}
    labels = {}

    for name, (nmi_df, genetic_dict, label) in systems_data.items():
        all_bw_nmi[name] = map_to_bw_pairs(nmi_df, genetic_dict)
        labels[name] = label

    pair_counts = defaultdict(int)
    for bw_nmi in all_bw_nmi.values():
        for pair in bw_nmi.keys():
            pair_counts[pair] += 1

    n_systems = len(all_bw_nmi)
    min_count = int(min_presence * n_systems)
    selected_pairs = sorted([p for p, c in pair_counts.items() if c >= min_count])

    print(f"Total unique BW pairs found: {len(pair_counts)}")
    print(f"Pairs in >= {min_presence*100:.0f}% of systems ({min_count}+ receptors): {len(selected_pairs)}")

    X = np.full((n_systems, len(selected_pairs)), np.nan)
    sample_names = []
    y = []

    for i, (name, bw_nmi) in enumerate(all_bw_nmi.items()):
        sample_names.append(name)
        y.append(labels[name])
        for j, pair in enumerate(selected_pairs):
            if pair in bw_nmi:
                X[i, j] = bw_nmi[pair]

    feature_names = [f"{p[0]}_{p[1]}" for p in selected_pairs]

    missing_pct = 100 * np.isnan(X).sum() / X.size
    print(f"Missing values: {missing_pct:.1f}%")

    return X, np.array(y), feature_names, sample_names

def get_region(bw):
    """Extract structural region from BW number."""
    try:
        tm = int(float(bw))
        return f"TM{tm}" if tm <= 7 else "H8"
    except:
        return bw


def add_region_features(X, feature_names):
    """Add mean NMI per region pair."""
    pair_regions = []
    for fn in feature_names:
        parts = fn.split('_')
        r1, r2 = get_region(parts[0]), get_region(parts[1])
        pair_regions.append(tuple(sorted([r1, r2])))

    unique_regions = sorted(set(pair_regions))
    region_features = np.zeros((X.shape[0], len(unique_regions)))

    for i, rp in enumerate(unique_regions):
        mask = [pr == rp for pr in pair_regions]
        if sum(mask) > 0:
            region_features[:, i] = np.nanmean(X[:, mask], axis=1)

    region_names = [f"region_{rp[0]}_{rp[1]}" for rp in unique_regions]

    print(f"Added {len(region_names)} region-aggregated features")
    print(f"Region features: {region_names}")

    return region_features, region_names

def compare_classifiers(X, y, cv_splits=5, random_state=42):
    """Nested CV (outer 5-fold, inner 3-fold GridSearchCV), one copy of each pipeline per label."""
    le = LABELS
    y_enc = np.asarray(y)

    outer_cv = LabelSetStratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    inner_cv = LabelSetStratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    n_features = min(20, X.shape[1])

    pipelines = {
        'Random Forest': Pipeline([
            ('selector', SelectKBest(f_classif, k=n_features)),
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=random_state))
        ]),
        'Gradient Boosting': Pipeline([
            ('selector', SelectKBest(f_classif, k=n_features)),
            ('clf', GradientBoostingClassifier(random_state=random_state))
        ]),
        'Logistic Regression': Pipeline([
            ('selector', SelectKBest(f_classif, k=n_features)),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state))
        ]),
        'SVM (RBF)': Pipeline([
            ('selector', SelectKBest(f_classif, k=n_features)),
            ('clf', SVC(kernel='rbf', class_weight='balanced', random_state=random_state))
        ]),
        'SVM (Linear)': Pipeline([
            ('selector', SelectKBest(f_classif, k=n_features)),
            ('clf', SVC(kernel='linear', class_weight='balanced', random_state=random_state))
        ])
    }

    param_grids = {
        'Random Forest': {
            'selector__k': [10, 15, 20],
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [2, 3, 4, 5],
            'clf__min_samples_leaf': [2, 3, 5]
        },
        'Gradient Boosting': {
            'selector__k': [10, 15, 20],
            'clf__n_estimators': [50, 100, 150],
            'clf__max_depth': [2, 3, 4],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__min_samples_leaf': [2, 3, 5]
        },
        'Logistic Regression': {
            'selector__k': [10, 15, 20],
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__penalty': ['l2']
        },
        'SVM (RBF)': {
            'selector__k': [10, 15, 20],
            'clf__C': [0.1, 1.0, 10.0],
            'clf__gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'SVM (Linear)': {
            'selector__k': [10, 15, 20],
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }
    }

    pipelines = {name: MultiOutputClassifier(pipe) for name, pipe in pipelines.items()}
    param_grids = {name: {f'estimator__{k}': v for k, v in grid.items()} for name, grid in param_grids.items()}

    results = {}
    best_params = {}

    print("="*70)
    print("Classifier Comparison with Hyperparameter Optimization")
    print("(Nested CV: outer=5-fold, inner=3-fold GridSearchCV)")
    print("="*70)

    for name, pipe in pipelines.items():
        print(f"\n{name}:")
        print(f"  Tuning hyperparameters...")

        grid_search = GridSearchCV(
            pipe,
            param_grids[name],
            cv=inner_cv,
            scoring=macro_bacc_scorer,
            n_jobs=-1,
            refit=True
        )

        outer_scores = cross_val_score(
            grid_search, X, y_enc,
            cv=outer_cv,
            scoring=macro_bacc_scorer
        )

        grid_search.fit(X, y_enc)
        best_params[name] = grid_search.best_params_

        results[name] = {
            'mean': outer_scores.mean(),
            'std': outer_scores.std(),
            'scores': outer_scores,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_
        }

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Inner CV score: {grid_search.best_score_:.3f}")
        print(f"  Outer CV score: {outer_scores.mean():.3f} (+/- {outer_scores.std():.3f})")

    best = max(results.items(), key=lambda x: x[1]['mean'])
    print("\n" + "="*70)
    print(f"Best: {best[0]} ({best[1]['mean']:.3f})")
    print("="*70)

    return results, le, best_params

def leave_one_out_evaluation(X, y, sample_names, n_features=15):
    """LOO-CV of the fixed Random Forest, one copy per label."""
    le = LABELS
    y_enc = np.asarray(y)

    n_features = min(n_features, X.shape[1])

    pipeline = MultiOutputClassifier(Pipeline([
        ('selector', SelectKBest(f_classif, k=n_features)),
        ('clf', RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3,
            class_weight='balanced', random_state=42
        ))
    ]))

    print(f"Running Leave-One-Out CV on {len(y)} samples...")
    predictions = cross_val_predict(pipeline, X, y_enc, cv=LeaveOneOut())

    bal_acc = macro_balanced_accuracy(y_enc, predictions)

    print(f"\nLOO-CV Balanced Accuracy: {bal_acc:.3f}")
    for j, lab in enumerate(LABELS):
        print(f"  {lab}: {balanced_accuracy_score(y_enc[:, j], predictions[:, j]):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_enc, predictions, target_names=LABELS, zero_division=0))

    mcm = multilabel_confusion_matrix(y_enc, predictions)

    fig, axes = plt.subplots(2, len(LABELS), figsize=(15, 9))

    for j, lab in enumerate(LABELS):
        cm = mcm[j]
        ticks = [f'no {lab}', lab]

        ax = axes[0, j]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=ticks, yticklabels=ticks, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{lab}: LOO-CV Confusion Matrix\nBalanced Accuracy: {balanced_accuracy_score(y_enc[:, j], predictions[:, j]):.3f}')

        ax = axes[1, j]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=ticks, yticklabels=ticks, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized (Row %)')

    plt.suptitle(f'LOO-CV, coupling-map labels: mean Balanced Accuracy {bal_acc:.3f}')
    plt.tight_layout()
    plt.savefig('loo_confusion_matrix_multilabel.png', dpi=300)
    plt.show()

    return predictions, y_enc, le, bal_acc


def analyze_feature_importance(X_scaled, y, feature_names, top_n_select=50, top_n_plot=25, save_results=True):
    """Rank features by mean ANOVA F, Gini, permutation importance and per-label mean differences."""
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    y_enc = np.asarray(y)

    f_scores, p_values = f_multilabel(X_scaled, y_enc)
    fscore_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)

    fscore_df['type'] = fscore_df['feature'].apply(
        lambda x: 'Region' if x.startswith('region_') else 'BW Pair'
    )

    print(f"Top 15 features by F-score:")
    print(fscore_df.head(15)[['feature', 'f_score', 'p_value']].to_string(index=False))

    n_select = min(top_n_select, X_scaled.shape[1])
    selector = SelectKBest(f_multilabel, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y_enc)

    selected_mask = selector.get_support()
    selected_feature_names = [f for f, m in zip(feature_names, selected_mask) if m]

    print(f"Training RF on top {n_select} features...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_selected, y_enc)

    gini_df = pd.DataFrame({
        'feature': selected_feature_names,
        'gini_importance': rf.feature_importances_
    }).sort_values('gini_importance', ascending=False)

    print(f"\nTop 15 features by Gini importance:")
    print(gini_df.head(15).to_string(index=False))

    perm_imp = permutation_importance(
        rf, X_selected, y_enc, n_repeats=30, random_state=42,
        scoring=macro_bacc_scorer, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'feature': selected_feature_names,
        'perm_importance': perm_imp.importances_mean,
        'std': perm_imp.importances_std
    }).sort_values('perm_importance', ascending=False)

    print(f"\nTop 15 features by Permutation importance:")
    print(perm_df.head(15)[['feature', 'perm_importance', 'std']].to_string(index=False))

    print("\n" + "-"*50)
    print("METHOD 4: Class Mean Differences")
    print("-"*50)

    #Per label: mean with the label minus mean without
    label_diffs = {lab: X_selected[y_enc[:, j] == 1].mean(axis=0) - X_selected[y_enc[:, j] == 0].mean(axis=0)
                   for j, lab in enumerate(LABELS)}

    diffs = []
    for i, feat in enumerate(selected_feature_names):
        values = [label_diffs[c][i] for c in LABELS]
        max_diff = max(abs(v) for v in values)
        diffs.append({
            'feature': feat,
            'max_diff': max_diff,
            **{f"{c}_diff": label_diffs[c][i] for c in LABELS}
        })

    diff_df = pd.DataFrame(diffs).sort_values('max_diff', ascending=False)

    print(f"\nTop 15 features by class separation:")
    print(diff_df.head(15).to_string(index=False))

    combined = pd.DataFrame({'feature': selected_feature_names})
    combined['f_score'] = combined['feature'].map(dict(zip(fscore_df['feature'], fscore_df['f_score'])))
    combined['gini'] = combined['feature'].map(dict(zip(gini_df['feature'], gini_df['gini_importance'])))
    combined['permutation'] = combined['feature'].map(dict(zip(perm_df['feature'], perm_df['perm_importance'])))
    combined['class_diff'] = combined['feature'].map(dict(zip(diff_df['feature'], diff_df['max_diff'])))

    for col in ['f_score', 'gini', 'permutation', 'class_diff']:
        vals = combined[col].dropna()
        combined[f'{col}_norm'] = (combined[col] - vals.min()) / (vals.max() - vals.min()) if len(vals) > 0 and vals.max() > vals.min() else 0

    combined['avg_importance'] = combined[[c for c in combined.columns if c.endswith('_norm')]].mean(axis=1)
    combined['type'] = combined['feature'].apply(lambda x: 'Region' if x.startswith('region_') else 'BW Pair')
    importance_df = combined.sort_values('avg_importance', ascending=False)

    print("\nTop 25 Features (Combined Ranking):")
    print(importance_df[['feature', 'avg_importance', 'f_score', 'gini', 'permutation', 'type']].head(25).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 10))
    top_features = importance_df.head(top_n_plot)
    colors = ['#e74c3c' if t == 'Region' else '#3498db' for t in top_features['type']]
    ax.barh(range(len(top_features)), top_features['avg_importance'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Combined Importance Score')
    ax.set_title(f'Top {top_n_plot} Features for Classification')
    legend_elements = [Patch(facecolor='#3498db', label='BW Pair'),
                       Patch(facecolor='#e74c3c', label='Region')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    if save_results:
        plt.savefig('feature_importance_combined_multilabel.png', dpi=300)
    plt.show()

    #Top 10 features by label
    top10_features = importance_df.head(10)['feature'].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(top10_features):
        ax = axes[idx]
        feat_idx = selected_feature_names.index(feat)
        data = [X_selected[y_enc[:, j] == v, feat_idx] for j in range(len(LABELS)) for v in (1, 0)]
        bp = ax.boxplot(data, labels=[f'{lab}{s}' for lab in LABELS for s in ('+', '-')], patch_artist=True)
        colors_box = ['#2ecc71', '#d5d8dc', '#e74c3c', '#d5d8dc', '#3498db', '#d5d8dc']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis='x', labelsize=8)

    plt.suptitle('Top 10 Features: Distribution by Label (+ with, - without the label)', fontsize=12)
    plt.tight_layout()
    if save_results:
        plt.savefig('feature_class_distributions_multilabel.png', dpi=300)
    plt.show()

    if save_results:
        importance_df.to_csv('feature_importance_results_multilabel.csv', index=False)
        print("\nSaved: feature_importance_results_multilabel.csv")
        print("Saved: feature_importance_combined_multilabel.png")
        print("Saved: feature_class_distributions_multilabel.png")

    return importance_df


#LOAD data

RECEPTOR_CONFIG = {
    'Gs': [
        ("../Gs/5HT7R_7XTC_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/AA2AR_5G53_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/ADRB3_9IJE_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/CNR1_8K8J_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/EDNRA_8XVI_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/EDNRB_8XVH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/FFAR4_8H4I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GP119_7WCM_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GP139_7VUH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GPBAR_9GYO_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GPR3_8WW2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GPR6_8TYW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/GPR12_7Y3G_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/NK1R_8U26_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/PE2R2_7CX2_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/PE2R4_8GDB_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/RXFP1_7TMW_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/SUCR1_8JPP_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/TAAR1_8W89_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gs/V2R_7DW9_DONE", "nmi_df.csv", "genetic_mapping.json")
    ],
    
    'Gi': [
        ("../Gi/AA1R_7LD4_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/AA3R_8X17_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/ADA2A_9CBL_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/CCR6_6WWZ_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/CNR1_9ERX_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/EDNRB_8IY5_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/FFAR1_9K1C_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/FFAR2_9K1D_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/MCHR1_8WWK_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/OPRK_8F7W_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/OPRM_8F7Q_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/PE2R4_8GCP_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/PTAFR_8XYD_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/5HT4R_7XTA_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/NMUR2_7XK8_PREPARED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/NTR1_6OS9_PREAPRED", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gi/PE2R3_8GDC_PREPARED", "nmi_df.csv", "genetic_mapping.json")
    ],
    
    'Gq': [
        ("../Gq/5HT2A_8UWL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/5HT2B_7SRR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/ADA1A_8THL_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/BKRB2_7F6I_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/EDNRA_8HCQ_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/EDNRB_8HCX_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/FFAR4_8G59_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/HRH2_8YN4_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/KISSR_8XGS_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/MCHR2_8WST_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/MRGX1_8JGF_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/MRGX2_7S8L_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/MTLR_8IBV_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/NMUR1_7W56_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/NMUR2_7W55_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/P2RY1_7XXH_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/PAR1_8XOR_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/PF2R_8IUK_DONE", "nmi_df.csv", "genetic_mapping.json"),
        ("../Gq/SSR2_7Y27_DONE", "nmi_df.csv", "genetic_mapping.json")
    ]
}

#Load MDPath data
systems_data = load_all_systems(RECEPTOR_CONFIG)

X, y, feature_names, sample_names = build_feature_matrix(systems_data, min_presence=0.5)

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nClass distribution:")
for cls, count in zip(*np.unique(y, return_counts=True)):
    print(f"  {cls}: {count}")

#Coupling-map families of each complex's receptor as targets; y keeps the bound G protein
Y = COUPLING.loc[sample_names, LABELS].values.astype(int)
print(f"\nCoupling-map label prevalence:")
for j, lab in enumerate(LABELS):
    print(f"  {lab}: {Y[:, j].sum()}/{len(Y)}")
print(f"Label combinations: {COUPLING.loc[sample_names, 'labels'].value_counts().to_dict()}")

X_region, region_names = add_region_features(X, feature_names)

X_combined = np.hstack([X, X_region])
all_feature_names = feature_names + region_names

print(f"Combined feature matrix: {X_combined.shape}")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_combined)

variances = np.var(X_imputed, axis=0)
var_threshold = 1e-10
keep_mask = variances > var_threshold

X_filtered = X_imputed[:, keep_mask]
filtered_feature_names = [f for f, k in zip(all_feature_names, keep_mask) if k]

print(f"Features after variance filter: {X_filtered.shape[1]} (removed {sum(~keep_mask)} constant features)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

print(f"\nFinal feature matrix: {X_scaled.shape}")

#Compare classifiers
clf_results, le, best_params = compare_classifiers(X_scaled, Y)

print("\n" + "="*70)
print("BEST HYPERPARAMETERS SUMMARY")
print("="*70)
for clf_name, params in best_params.items():
    print(f"\n{clf_name}:")
    for param, value in params.items():
        print(f"  {param}: {value}")

#Make some figures for SI
fig, ax = plt.subplots(figsize=(10, 6))

names = list(clf_results.keys())
means = [clf_results[n]['mean'] for n in names]
stds = [clf_results[n]['std'] for n in names]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
bars = ax.barh(names, means, xerr=stds, color=colors, alpha=0.8, capsize=5)
ax.set_xlabel('Balanced Accuracy (mean over Gi, Gq, Gs labels)')
ax.set_title('Classifier Comparison (5-Fold Stratified CV)')
ax.axvline(0.5, color='red', linestyle='--', label='Random guess (binary labels)')
ax.set_xlim(0, 1)
ax.legend()

for bar, mean in zip(bars, means):
    ax.text(mean + 0.05, bar.get_y() + bar.get_height()/2,
            f'{mean:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('classifier_comparison_multilabel.png', dpi=300)
plt.show()

#Leave one out
loo_preds, y_encoded, le, loo_accuracy = leave_one_out_evaluation(X_scaled, Y, sample_names)

#Feature importance
importance_df = analyze_feature_importance(
    X_scaled=X_scaled,
    y=Y,
    feature_names=filtered_feature_names
)
