# ============================================================
# feature_extractor.py
# STRICT TRAINING-ALIGNED FEATURE EXTRACTION
# ECFP + MACCS + TRAINING DESCRIPTORS + LOCAL ProtBERT
# Production Django Version (Stable)
# ============================================================

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from transformers import AutoTokenizer, AutoModel
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ============================================================
# CONFIGURATION
# ============================================================

ECFP_BITS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROTBERT_PATH = BASE_DIR / "artifacts" / "protbert_local"
TRAIN_DATA_PATH = BASE_DIR / "artifacts" / "MOA_features_with_descriptors.csv"


# ============================================================
# LOAD TRAINING FEATURE STRUCTURE (CRITICAL)
# ============================================================

print("🔍 Loading training feature structure...")

train_df = pd.read_csv(TRAIN_DATA_PATH, nrows=1)

exclude_cols = ["SMILES", "protein_sequence", "MOA", "MOA_encoded"]

ecfp_cols = [c for c in train_df.columns if c.startswith("ECFP_")]
maccs_cols = [c for c in train_df.columns if c.startswith("MACCS_")]
prot_cols = [c for c in train_df.columns if c.startswith("PROT_")]

descriptor_cols = [
    c for c in train_df.columns
    if c not in exclude_cols
    and not c.startswith(("ECFP_", "MACCS_", "PROT_"))
]

print(f"ECFP features: {len(ecfp_cols)}")
print(f"MACCS features: {len(maccs_cols)}")
print(f"PROT features: {len(prot_cols)}")
print(f"Descriptor features (training-aligned): {len(descriptor_cols)}")
print("Feature structure locked.\n")


# ============================================================
# LOAD MODELS (Once at Startup)
# ============================================================

print("🔬 Loading LOCAL ProtBERT...")
print(f"Device: {DEVICE}")

morgan_gen = GetMorganGenerator(radius=2, fpSize=ECFP_BITS)

tokenizer = AutoTokenizer.from_pretrained(PROTBERT_PATH)
protein_model = AutoModel.from_pretrained(PROTBERT_PATH).to(DEVICE)
protein_model.eval()

print("✅ ProtBERT loaded successfully\n")


# ============================================================
# SMILES → ECFP
# ============================================================

def smiles_to_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((ECFP_BITS,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ============================================================
# SMILES → MACCS
# ============================================================

def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ============================================================
# SMILES → TRAINING-ALIGNED DESCRIPTORS
# ============================================================

# Build descriptor lookup dictionary
descriptor_function_map = dict(Descriptors.descList)

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    values = []

    for name in descriptor_cols:
        if name in descriptor_function_map:
            try:
                values.append(float(descriptor_function_map[name](mol)))
            except:
                values.append(0.0)
        else:
            # Descriptor name not found (safety fallback)
            values.append(0.0)

    return np.array(values, dtype=float)


# ============================================================
# Protein → ProtBERT Embedding
# ============================================================

def protein_to_embedding(sequence, max_len=1024):

    if not sequence or not isinstance(sequence, str):
        raise ValueError("Invalid protein sequence")

    sequence = sequence[:max_len]
    sequence = " ".join(list(sequence))

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = protein_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy().flatten()


# ============================================================
# MASTER FEATURE EXTRACTION
# ============================================================

def extract_features(smiles, protein_sequence):
    """
    Returns features aligned EXACTLY with training dataset.
    """

    return {
        "ecfp": smiles_to_ecfp(smiles),
        "maccs": smiles_to_maccs(smiles),
        "descriptors": smiles_to_descriptors(smiles),
        "protein_embedding": protein_to_embedding(protein_sequence)
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

extract_protein_embedding = protein_to_embedding