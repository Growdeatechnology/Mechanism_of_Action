# ============================================================
# shift_analysis_engine.py
# INDUSTRY-GRADE MOA SHIFT + COUNTERFACTUAL ENGINE
# Fully aligned with training + perturbation engine
# ============================================================

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .model_loader import registry
from .ligand_perturbation_engine import generate_perturbations


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "artifacts/MOA_features_with_descriptors.csv"

print("\n=========== SHIFT ENGINE INITIALIZATION ===========")

try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded:", df.shape)
except Exception as e:
    df = None
    print("Dataset not found.", str(e))


GROUP_MAPPING = {
    "Inhibitor": "Inhibitory",
    "Antagonist": "Inhibitory",
    "Blocker": "Inhibitory",
    "Agonist": "Activating",
    "Activator": "Activating",
    "Stimulator": "Activating",
    "Modulator": "Modulatory",
    "Modulator (allosteric modulator)": "Modulatory",
    "Binder": "Binding"
}

if df is not None and "MOA" in df.columns:
    df["MOA_grouped"] = df["MOA"].map(GROUP_MAPPING)

classes = list(registry.label_mapping.values())


# ============================================================
# PRECOMPUTE FINGERPRINTS
# ============================================================

fps = []

if df is not None:
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, registry.FP_SIZE
            )
            fps.append(fp)
        else:
            fps.append(None)

print("Fingerprint generation complete.")
print("====================================================\n")


# ============================================================
# TRAINING-ALIGNED PREDICTION
# ============================================================

def predict_moa_from_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not hasattr(registry, "compute_features"):
        return None

    X_full = registry.compute_features(smiles)

    X_var = registry.selector.transform(X_full)

    var_feature_names = np.array(registry.feature_columns)[
        registry.selector.get_support()
    ]

    df_var = pd.DataFrame(X_var, columns=var_feature_names)
    X_selected = df_var[registry.selected_features]

    proba = registry.model.predict_proba(X_selected)[0]

    return {
        registry.label_mapping[i]: float(p)
        for i, p in enumerate(proba)
    }

# ========================================================
# INDUSTRY-GRADE FRAGMENT DIFFERENCE DETECTOR
# Extract real fragment SMILES using Morgan bitInfo
# ========================================================

def get_fragments_from_morgan(mol, radius=2, nBits=2048):

    if mol is None:
        return {}

    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nBits,
        bitInfo=bitInfo
    )

    fragments = {}

    for bit_id, atom_infos in bitInfo.items():

        frag_smiles_list = []

        for atom_idx, rad in atom_infos:

            env = Chem.FindAtomEnvironmentOfRadiusN(
                mol,
                rad,
                atom_idx
            )

            atoms = set()
            for bond_id in env:
                bond = mol.GetBondWithIdx(bond_id)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())

            if not atoms:
                continue

            submol = Chem.PathToSubmol(mol, env)

            frag_smiles = Chem.MolToSmiles(
                submol,
                canonical=True
            )

            frag_smiles_list.append(frag_smiles)

        if frag_smiles_list:
            fragments[bit_id] = list(set(frag_smiles_list))

    return fragments


def extract_fragment_changes(original_smiles, modified_smiles):

    mol1 = Chem.MolFromSmiles(original_smiles)
    mol2 = Chem.MolFromSmiles(modified_smiles)

    if mol1 is None or mol2 is None:
        return {"added": [], "removed": [], "replaced": []}

    frags1 = get_fragments_from_morgan(
        mol1,
        radius=2,
        nBits=registry.FP_SIZE
    )

    frags2 = get_fragments_from_morgan(
        mol2,
        radius=2,
        nBits=registry.FP_SIZE
    )

    bits1 = set(frags1.keys())
    bits2 = set(frags2.keys())

    removed_bits = bits1 - bits2
    added_bits = bits2 - bits1
    common_bits = bits1 & bits2

    added = []
    removed = []
    replaced = []

    # ----------------------------------------------------
    # Added fragments (new only)
    # ----------------------------------------------------
    for bit in added_bits:
        for frag in frags2[bit]:
            added.append(frag)

    # ----------------------------------------------------
    # Removed fragments (old only)
    # ----------------------------------------------------
    for bit in removed_bits:
        for frag in frags1[bit]:
            removed.append(frag)

    # ----------------------------------------------------
    # Replaced fragments (old → new pairs)
    # Same bit but different structures
    # ----------------------------------------------------
    for bit in common_bits:

        set1 = set(frags1[bit])
        set2 = set(frags2[bit])

        if set1 != set2:

            # fragments that disappeared
            old_only = set1 - set2

            # fragments that appeared
            new_only = set2 - set1

            for old_frag in old_only:
                for new_frag in new_only:
                    replaced.append({
                        "old": old_frag,
                        "new": new_frag
                    })

    # ----------------------------------------------------
    # Clean duplicates
    # ----------------------------------------------------
    added = list(set(added))[:5]
    removed = list(set(removed))[:5]

    # remove duplicate replacement pairs
    unique_pairs = []
    seen_pairs = set()

    for pair in replaced:
        key = (pair["old"], pair["new"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(pair)

    replaced = unique_pairs[:5]

    return {
        "added": added,
        "removed": removed,
        "replaced": replaced
    }


# ============================================================
# CORE SHIFT FUNCTION
# ============================================================

def moa_shift_trend(query_smiles, top_k=80, bins=6):

    if registry.model is None:
        return {"error": "Model not loaded."}

    if df is None:
        return {"error": "Dataset not loaded."}

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return {"error": "Invalid SMILES"}

    # ========================================================
    # SIMILARITY SEARCH
    # ========================================================

    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        query_mol, 2, registry.FP_SIZE
    )

    similarities = []

    for i, fp in enumerate(fps):
        if fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_hits = similarities[:top_k]

    trend_data = []

    for idx, sim in top_hits:

        row = df.iloc[idx]

        X_full = row[registry.feature_columns].values.reshape(1, -1)
        X_var = registry.selector.transform(X_full)

        var_feature_names = np.array(registry.feature_columns)[
            registry.selector.get_support()
        ]

        df_var = pd.DataFrame(X_var, columns=var_feature_names)
        X_selected = df_var[registry.selected_features]

        proba = registry.model.predict_proba(X_selected)[0]

        entry = {
            "Similarity": float(sim),
            "True_MOA": row.get("MOA_grouped", None)
        }

        for i, p in enumerate(proba):
            entry[registry.label_mapping[i]] = float(p)

        trend_data.append(entry)

    trend_df = pd.DataFrame(trend_data)

    if trend_df.empty:
        return {"error": "No valid neighbors found."}

    # ========================================================
    # BINNING (SAFE)
    # ========================================================

    try:
        trend_df["Similarity_Bin"] = pd.qcut(
            trend_df["Similarity"],
            q=bins,
            duplicates="drop"
        )
    except:
        trend_df["Similarity_Bin"] = "Single Bin"

    # ========================================================
    # TRUE DISTRIBUTION
    # ========================================================

    true_dist = (
        trend_df
        .groupby("Similarity_Bin")["True_MOA"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # ========================================================
    # MEAN PROBABILITIES
    # ========================================================

    mean_probs = (
        trend_df
        .groupby("Similarity_Bin")[classes]
        .mean()
    )

    # ========================================================
    # DRIFT SLOPES
    # ========================================================

    slopes = {}
    x = trend_df["Similarity"].values

    for cls in classes:
        y = trend_df[cls].values
        if len(np.unique(x)) > 1:
            slopes[cls] = float(np.polyfit(x, y, 1)[0])
        else:
            slopes[cls] = 0.0

    # ========================================================
    # QUERY PREDICTION
    # ========================================================

    query_prediction = predict_moa_from_smiles(query_smiles)

    # ========================================================
    # COUNTERFACTUAL ANALYSIS
    # ========================================================

    structural_shifts = []
    strongest_driver = None

    if query_prediction is not None:

        perturbations = generate_perturbations(query_smiles)

        for item in perturbations:

            mod_smiles = item["SMILES"]
            fragment_changes = extract_fragment_changes(query_smiles, mod_smiles)
            mod_prediction = predict_moa_from_smiles(mod_smiles)

            if mod_prediction is None:
                continue

            delta = {}
            impact_score = 0.0

            for cls in classes:
                delta_value = (
                    mod_prediction.get(cls, 0.0)
                    - query_prediction.get(cls, 0.0)
                )
                delta[cls] = float(delta_value)
                impact_score += delta_value ** 2

            impact_score = float(np.sqrt(impact_score))

            if impact_score < 0.001:
                continue                

            # ----------------------------------------------------
            # Determine dominant mechanistic shift
            # ----------------------------------------------------

            max_cls = max(delta, key=lambda k: abs(delta[k]))
            delta_value = delta[max_cls]

            if delta_value > 0:
                dominant_effect = f"Mechanistic shift toward {max_cls}"
            else:
                dominant_effect = f"Mechanistic shift away from {max_cls}"

            # Format replacement pairs for UI
            formatted_replacements = []

            for pair in fragment_changes["replaced"]:
                formatted_replacements.append(
                    f'{pair["old"]} → {pair["new"]}'
                )


            structural_shifts.append({
                "Modification_Type": item["Modification_Type"],   # 👈 matches template
                "Modified_SMILES": mod_smiles,
                "Delta_Shift": delta,
                "Impact_Score": impact_score,
                "Primary_Direction": dominant_effect,

                # UI-ready fragment fields
                "Fragments_Added": fragment_changes["added"],
                "Fragments_Removed": fragment_changes["removed"],
                "Fragments_Replaced": formatted_replacements,

                # Optional label
                "Delta_Label": item["Delta_Label"]
            })

        # Remove duplicate delta patterns
        unique_patterns = {}
        for item in structural_shifts:
            key = tuple(round(v, 4) for v in item["Delta_Shift"].values())
            if key not in unique_patterns:
                unique_patterns[key] = item

        structural_shifts = list(unique_patterns.values())

        structural_shifts.sort(
            key=lambda x: x["Impact_Score"],
            reverse=True
        )

        structural_shifts = structural_shifts[:10]
        strongest_driver = structural_shifts[0] if structural_shifts else None

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    return {
        "Query_Prediction": query_prediction,
        "Similarity_Trend_Table": trend_df.to_dict(orient="records"),
        "True_MOA_Distribution": true_dist.round(3).to_dict(),
        "Mean_Probability_Per_Bin": mean_probs.round(3).to_dict(),
        "Drift_Slopes": slopes,
        "Structural_Counterfactual_Shifts": structural_shifts,
        "Strongest_Driver": strongest_driver
    }


# ============================================================
# DJANGO SERVICE WRAPPER
# ============================================================

class ShiftAnalysisEngine:

    def analyze(self, smiles, top_k=80, bins=6):
        return moa_shift_trend(smiles, top_k=top_k, bins=bins)