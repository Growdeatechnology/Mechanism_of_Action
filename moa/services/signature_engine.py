# ============================================================
# signature_engine.py
# 2.4 MOA-Specific Chemical Signature Discovery
# Django Production Version (Service Wrapped)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PathToSubmol

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "MOA_features_with_descriptors.csv"

RADIUS = 2
N_BITS = 2048
TOP_K = 15

# ============================================================
# LOAD DATA (ONCE)
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    print(f"[SignatureEngine] Dataset loading failed: {e}")
    df = None

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
    df = df.dropna(subset=["MOA_grouped"]).reset_index(drop=True)

# ============================================================
# PRECOMPUTE FINGERPRINT MATRIX
# ============================================================

bit_matrix = []
bit_info_store = []

if df is not None:

    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            bit_matrix.append(np.zeros(N_BITS))
            bit_info_store.append({})
            continue

        bit_info = {}

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=RADIUS,
            nBits=N_BITS,
            bitInfo=bit_info
        )

        arr = np.zeros((N_BITS,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)

        bit_matrix.append(arr)
        bit_info_store.append(bit_info)

    bit_matrix = np.array(bit_matrix)

# ============================================================
# ENRICHMENT
# ============================================================

def compute_enrichment(bit_matrix, labels, class_name):

    class_mask = labels == class_name
    other_mask = labels != class_name

    if class_mask.sum() == 0:
        return np.zeros(bit_matrix.shape[1])

    p_class = bit_matrix[class_mask].mean(axis=0)
    p_other = bit_matrix[other_mask].mean(axis=0)

    return p_class - p_other

# ============================================================
# FRAGMENT EXTRACTION
# ============================================================

def extract_fragment(mol, bit_info, bit):

    if bit not in bit_info:
        return None

    for atom_idx, radius in bit_info[bit]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        if not env:
            continue
        submol = PathToSubmol(mol, env)
        return Chem.MolToSmiles(submol)

    return None

# ============================================================
# CORE FUNCTION
# ============================================================

def compute_moa_signatures(top_k=TOP_K):

    if df is None:
        return {"error": "Dataset not loaded."}

    labels = df["MOA_grouped"].values
    unique_classes = sorted(df["MOA_grouped"].unique())

    signatures = {}

    for cls in unique_classes:

        enrichment = compute_enrichment(bit_matrix, labels, cls)
        top_bits = np.argsort(enrichment)[::-1][:top_k]

        fragments = []
        class_indices = np.where(labels == cls)[0]

        for bit in top_bits:

            for idx in class_indices:
                if bit_matrix[idx][bit] == 1:

                    mol = Chem.MolFromSmiles(df.iloc[idx]["SMILES"])
                    frag = extract_fragment(mol, bit_info_store[idx], bit)

                    if frag:
                        fragments.append({
                            "Fragment": frag,
                            "Enrichment_Score": round(float(enrichment[bit]), 4)
                        })
                        break

        signatures[cls] = fragments

    return signatures

# ============================================================
# DJANGO SERVICE WRAPPER (IMPROVED)
# ============================================================

class SignatureEngine:
    """
    Service wrapper used by views.py
    """

    def __init__(self):
        pass

    def _clean_signatures(self, raw_signatures, top_k=TOP_K):
        """
        Deduplicate fragments, keep highest enrichment score,
        sort descending, and limit to top_k.
        """

        cleaned_signatures = {}

        for cls, frags in raw_signatures.items():

            unique = {}

            for f in frags:
                frag = f.get("Fragment")
                score = f.get("Enrichment_Score", 0)

                # Keep highest enrichment for duplicate fragments
                if frag not in unique or score > unique[frag]:
                    unique[frag] = score

            # Convert back to list
            cleaned_list = [
                {"Fragment": k, "Enrichment_Score": v}
                for k, v in unique.items()
            ]

            # Sort descending
            cleaned_list = sorted(
                cleaned_list,
                key=lambda x: x["Enrichment_Score"],
                reverse=True
            )

            # Keep top_k after cleaning
            cleaned_signatures[cls] = cleaned_list[:top_k]

        return cleaned_signatures

    def get_signatures(self, top_k=TOP_K, clean=True):
        """
        Returns chemical signatures.
        If clean=True → returns deduplicated & sorted signatures.
        """

        raw = compute_moa_signatures(top_k=top_k)

        if isinstance(raw, dict) and "error" in raw:
            return raw

        if clean:
            return self._clean_signatures(raw, top_k=top_k)

        return raw