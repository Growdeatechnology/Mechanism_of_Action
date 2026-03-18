# ============================================================
# similarity_engine.py
# 2.3 Multimodal Similarity Engine (Ligand + Protein)
# Registry-Aligned | No Duplicates | Production Version
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity

from .model_loader import registry


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "MOA_features_with_descriptors.csv"

RADIUS = 2
N_BITS = 2048

LIGAND_WEIGHT = 0.6
PROTEIN_WEIGHT = 0.4


# ============================================================
# LOAD DATA
# ============================================================

df = None

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"[SimilarityEngine] Dataset loading failed: {e}")


# ============================================================
# GROUPING
# ============================================================

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

if df is not None:
    df["MOA_grouped"] = df["MOA"].map(GROUP_MAPPING)
    df = df.dropna(subset=["MOA_grouped"]).reset_index(drop=True)


# ============================================================
# FEATURE GROUPS
# ============================================================

if df is not None:
    ecfp_cols = [c for c in df.columns if c.startswith("ECFP_")]
    maccs_cols = [c for c in df.columns if c.startswith("MACCS_")]
    prot_cols = [c for c in df.columns if c.startswith("PROT_")]

    dataset_prot_matrix = df[prot_cols].values
else:
    ecfp_cols, maccs_cols, prot_cols = [], [], []
    dataset_prot_matrix = None


# ============================================================
# PRECOMPUTE ECFP FINGERPRINTS
# ============================================================

dataset_fps = []

if df is not None:
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=RADIUS,
                nBits=N_BITS
            )
            dataset_fps.append(fp)
        else:
            dataset_fps.append(None)


# ============================================================
# CORE MULTIMODAL FUNCTION
# ============================================================

def find_multimodal_similarity(
    df,
    dataset_prot_matrix,
    dataset_fps,
    query_smiles,
    query_prot_vector=None,
    top_k=20
):

    if df is None:
        return {"error": "Dataset not loaded."}

    if registry.model is None:
        return {"error": "Model not loaded."}

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return {"error": "Invalid SMILES"}

    # -------------------------
    # Ligand fingerprint
    # -------------------------
    query_fp = AllChem.GetMorganFingerprintAsBitVect(
        query_mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    # -------------------------
    # Protein similarity vector
    # -------------------------
    if query_prot_vector is not None and dataset_prot_matrix is not None:

        protein_sim_vector = cosine_similarity(
            query_prot_vector.reshape(1, -1),
            dataset_prot_matrix
        )[0]

        # Normalize cosine [-1,1] → [0,1]
        protein_sim_vector = (protein_sim_vector + 1) / 2

    else:
        protein_sim_vector = np.zeros(len(df))

    # -------------------------
    # Compute raw similarity vectors
    # -------------------------

    ligand_sims = []
    protein_sims = []
    valid_indices = []

    for i, fp in enumerate(dataset_fps):

        if fp is None:
            continue

        ligand_sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        protein_sim = protein_sim_vector[i]

        ligand_sims.append(ligand_sim)
        protein_sims.append(protein_sim)
        valid_indices.append(i)

    ligand_sims = np.array(ligand_sims)
    protein_sims = np.array(protein_sims)


    if query_prot_vector is not None:

        mean_lig = ligand_sims.mean()
        std_lig = ligand_sims.std() + 1e-8

        mean_prot = protein_sims.mean()
        std_prot = protein_sims.std() + 1e-8

        ligand_z = (ligand_sims - mean_lig) / std_lig
        protein_z = (protein_sims - mean_prot) / std_prot

        combined_raw = (
            LIGAND_WEIGHT * ligand_z +
            PROTEIN_WEIGHT * protein_z
        )

        # Convert to 0–1 range (sigmoid) (UI friendly)
        combined_scores = 1 / (1 + np.exp(-combined_raw))

    else:
        combined_scores = ligand_sims


    # -------------------------
    # Build similarity tuples and sort
    # -------------------------
    similarities = []

    for idx, lig, prot, comb in zip(
            valid_indices,
            ligand_sims,
            protein_sims,
            combined_scores):

        similarities.append((idx, lig, prot, comb))

    similarities.sort(key=lambda x: x[3], reverse=True)

    # -------------------------
    # Remove duplicates (by SMILES)
    # -------------------------
    seen_pairs = set()
    filtered_hits = []

    for idx, ligand_sim, protein_sim, combined_score in similarities:

        row = df.iloc[idx]
        smiles = row["SMILES"]
        protein_seq = row["protein_sequence"]

        pair_key = (smiles, protein_seq)

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            filtered_hits.append((idx, ligand_sim, protein_sim, combined_score))

        if len(filtered_hits) >= top_k:
            break

    # -------------------------
    # Model inference
    # -------------------------
    feature_columns = registry.feature_columns
    selected_features = registry.selected_features

    results = []

    for idx, ligand_sim, protein_sim, combined_score in filtered_hits:

        row = df.iloc[idx]

        X = row[feature_columns].values.astype(float).reshape(1, -1)

        X_var = registry.selector.transform(X)
        var_feature_names = np.array(feature_columns)[
            registry.selector.get_support()
        ]

        df_var = pd.DataFrame(X_var, columns=var_feature_names)
        df_selected = df_var[selected_features]

        proba = registry.model.predict_proba(df_selected)[0]

        result_entry = {
            "Combined_Similarity": round(float(combined_score), 4),
            "Ligand_Similarity": round(float(ligand_sim), 4),
            "Protein_Similarity": round(float(protein_sim), 4),
            "SMILES": row["SMILES"],
            "Protein_Sequence": row.get("protein_sequence", ""),
            "True_MOA": row["MOA_grouped"]
        }

        for i, p in enumerate(proba):
            label = registry.label_mapping.get(i, str(i))
            result_entry[label] = round(float(p), 4)

        results.append(result_entry)

    return {
        "Query_SMILES": query_smiles,
        "Results": results
    }


# ============================================================
# DJANGO SERVICE WRAPPER
# ============================================================

class SimilarityEngine:

    def __init__(self):

        if df is None:
            raise RuntimeError("Dataset not loaded.")

        if registry.model is None:
            raise RuntimeError("Model not loaded.")

        self.df = df

        self.prot_cols = [c for c in self.df.columns if c.startswith("PROT_")]

        if len(self.prot_cols) == 0:
            raise RuntimeError("No PROT columns found in dataset.")

        self.dataset_prot_matrix = self.df[self.prot_cols].values
        self.dataset_fps = dataset_fps  # reuse precomputed list


    def search(self, smiles, prot_vector=None, top_k=20):

        return find_multimodal_similarity(
            df=self.df,
            dataset_prot_matrix=self.dataset_prot_matrix,
            dataset_fps=self.dataset_fps,
            query_smiles=smiles,
            query_prot_vector=prot_vector,
            top_k=top_k
        )