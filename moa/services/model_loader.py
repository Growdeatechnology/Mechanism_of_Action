# ============================================================
# model_loader.py
# Central Artifact Registry for Entire Platform
# ============================================================

import joblib
import json
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:

    def __init__(self):

        print("\n========== LOADING MOA ARTIFACTS ==========")

        try:
            # ------------------------------------------------
            # 1️⃣ CatBoost Model
            # ------------------------------------------------
            self.model = joblib.load(
                ARTIFACTS / "moa_catboost_model_with_desc.joblib"
            )
            print("✔ Model loaded")

            # ------------------------------------------------
            # 2️⃣ Variance Selector
            # ------------------------------------------------
            self.selector = joblib.load(
                ARTIFACTS / "variance_selector.joblib"
            )
            print("✔ Variance selector loaded")

            # ------------------------------------------------
            # 3️⃣ Selected Feature Names
            # ------------------------------------------------
            with open(ARTIFACTS / "top_feature_names.json") as f:
                self.selected_features = json.load(f)
            print("✔ Selected feature list loaded")

            # ------------------------------------------------
            # 4️⃣ Label Mapping
            # ------------------------------------------------
            with open(ARTIFACTS / "moa_label_mapping.json") as f:
                label_map = json.load(f)

            self.label_mapping = {int(k): v for k, v in label_map.items()}
            print("✔ Label mapping loaded")

            # ------------------------------------------------
            # 5️⃣ Load Dataset
            # ------------------------------------------------
            dataset_path = DATA_DIR / "MOA_features_with_descriptors.csv"
            self.dataset = pd.read_csv(dataset_path)
            print("✔ Dataset loaded:", self.dataset.shape)


            # ------------------------------------------------
            # DEFINE FEATURE STRUCTURE (UPDATED FOR DESCRIPTORS)
            # ------------------------------------------------

            exclude_cols = ["SMILES", "protein_sequence", "MOA", "MOA_encoded"]

            self.feature_columns = [
                c for c in self.dataset.columns
                if c not in exclude_cols
            ]

            self.ecfp_cols = [c for c in self.feature_columns if c.startswith("ECFP_")]
            self.maccs_cols = [c for c in self.feature_columns if c.startswith("MACCS_")]
            self.prot_cols = [c for c in self.feature_columns if c.startswith("PROT_")]

            self.descriptor_columns = [
                c for c in self.feature_columns
                if not c.startswith(("ECFP_", "MACCS_", "PROT_"))
            ]


            self.ecfp_cols = [
                c for c in self.feature_columns if c.startswith("ECFP_")
            ]

            self.prot_cols = [
                c for c in self.feature_columns if c.startswith("PROT_")
            ]

            # Infer fingerprint size automatically
            self.FP_SIZE = len(self.ecfp_cols)

            print("✔ Feature column structure registered")

            # ------------------------------------------------
            # 7️⃣ ProtBERT (Lazy Load)
            # ------------------------------------------------
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.protein_tokenizer = None
            self.protein_model = None

            print("✔ Registry initialized successfully")

        except Exception as e:
            print("❌ Artifact loading failed:", e)

            self.model = None
            self.selector = None
            self.selected_features = []
            self.label_mapping = {}
            self.dataset = None

            self.feature_columns = []
            self.ecfp_cols = []
            self.prot_cols = []
            self.FP_SIZE = 0

            self.protein_tokenizer = None
            self.protein_model = None


    # ========================================================
    # LAZY LOAD PROTBERT
    # ========================================================
    def load_protbert(self):

        if self.protein_model is not None:
            return

        print("🔬 Loading LOCAL ProtBERT...")

        self.protein_tokenizer = AutoTokenizer.from_pretrained(
            ARTIFACTS / "protbert_local"
        )

        self.protein_model = AutoModel.from_pretrained(
            ARTIFACTS / "protbert_local"
        ).to(self.device)

        self.protein_model.eval()

        print("✅ ProtBERT loaded successfully")

    
    # ========================================================
    # FEATURE GENERATION (CRITICAL FOR INFERENCE)
    # ========================================================
    def compute_features(self, smiles):
        """
        Compute features EXACTLY aligned with training pipeline.
        Returns DataFrame with same feature_columns order.
        """

        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit.Chem import Descriptors
        import numpy as np

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        feature_dict = {}

        # ----------------------------------------------------
        # 1️⃣ ECFP (Morgan Fingerprint)
        # ----------------------------------------------------
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=self.FP_SIZE
        )

        ecfp_array = np.zeros((self.FP_SIZE,))
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, ecfp_array)

        for i, col in enumerate(self.ecfp_cols):
            feature_dict[col] = ecfp_array[i]

        # ----------------------------------------------------
        # 2️⃣ MACCS Keys
        # ----------------------------------------------------
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((len(self.maccs_cols),))
        DataStructs.ConvertToNumpyArray(maccs, maccs_array)

        for i, col in enumerate(self.maccs_cols):
            feature_dict[col] = maccs_array[i]

        # ----------------------------------------------------
        # 3️⃣ RDKit Descriptors
        # ----------------------------------------------------
        for col in self.descriptor_columns:
            try:
                desc_func = getattr(Descriptors, col)
                feature_dict[col] = desc_func(mol)
            except:
                feature_dict[col] = 0.0

        # ----------------------------------------------------
        # 4️⃣ PROTEIN FEATURES (Zero if absent)
        # ----------------------------------------------------
        for col in self.prot_cols:
            feature_dict[col] = 0.0

        # ----------------------------------------------------
        # ALIGN COLUMN ORDER EXACTLY
        # ----------------------------------------------------
        feature_vector = [feature_dict.get(c, 0.0)
                          for c in self.feature_columns]

        return pd.DataFrame([feature_vector],
                            columns=self.feature_columns)

# ============================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================

registry = ModelRegistry()