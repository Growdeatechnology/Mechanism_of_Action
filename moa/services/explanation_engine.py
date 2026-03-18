# ============================================================
# explanation_engine.py
# FULLY TRAINING-ALIGNED TARGET-AWARE MOA EXPLANATION ENGINE
# Supports:
# - ECFP
# - MACCS
# - RDKit Descriptors
# - ProtBERT
# ============================================================

import numpy as np
import pandas as pd
from catboost import Pool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger
from rdkit.Chem.MACCSkeys import smartsPatts as MACCS_SMARTS
from .model_loader import registry
from .feature_extractor import extract_protein_embedding, smiles_to_descriptors

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Fragment Extraction
# ============================================================

def extract_fragment_smiles(mol, atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env:
        return None

    submol = Chem.PathToSubmol(mol, env)
    return Chem.MolToSmiles(submol)


# ============================================================
# Explanation Engine
# ============================================================

class ExplanationEngine:

    def __init__(self):
        self.model = registry.model
        self.selector = registry.selector
        self.selected_features = registry.selected_features
        self.label_mapping = registry.label_mapping
        self.feature_columns = registry.feature_columns

        self.ecfp_cols = registry.ecfp_cols
        self.maccs_cols = registry.maccs_cols
        self.prot_cols = registry.prot_cols
        self.descriptor_cols = registry.descriptor_columns

        self.FP_SIZE = registry.FP_SIZE

    # ============================================================
    # Build Feature Vector
    # ============================================================

    def build_feature_dataframe(self, smiles, protein_vector):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None

        # ---- ECFP ----
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=self.FP_SIZE,
            bitInfo=bit_info,
        )

        ecfp_array = np.zeros((self.FP_SIZE,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, ecfp_array)

        # ---- MACCS ----
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((maccs_fp.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)

        # ---- DESCRIPTORS ----
        descriptor_array = smiles_to_descriptors(smiles)

        feature_dict = {}

        # Map ECFP
        for col in self.ecfp_cols:
            bit_index = int(col.split("_")[1])
            feature_dict[col] = int(ecfp_array[bit_index])

        # Map MACCS
        for col in self.maccs_cols:
            bit_index = int(col.split("_")[1])
            feature_dict[col] = int(maccs_array[bit_index])

        # Map PROT
        if len(protein_vector) != len(self.prot_cols):
            raise ValueError(
                f"Protein vector mismatch: expected {len(self.prot_cols)}, got {len(protein_vector)}"
            )

        for i, col in enumerate(self.prot_cols):
            feature_dict[col] = float(protein_vector[i])

        # Map DESCRIPTORS
        for name, value in zip(self.descriptor_cols, descriptor_array):
            feature_dict[name] = float(value)

        # ---- Build DataFrame aligned to training ----
        df_query_full = pd.DataFrame(
            [feature_dict],
            columns=self.feature_columns
        )

        # ---- Apply selector ----
        X_var = self.selector.transform(df_query_full.values)
        var_feature_names = np.array(self.feature_columns)[
            self.selector.get_support()
        ]

        df_var = pd.DataFrame(X_var, columns=var_feature_names)
        df_selected = df_var[self.selected_features]

        return mol, df_selected, bit_info

    # ============================================================
    # Explain
    # ============================================================

    def explain(self, smiles, protein_sequence):

        protein_vector = extract_protein_embedding(protein_sequence)

        mol, df_query, bit_info = self.build_feature_dataframe(
            smiles, protein_vector
        )

        if mol is None:
            return {"error": "Invalid SMILES"}

        # ================= Prediction =================
        proba = self.model.predict_proba(df_query)[0]
        pred_idx = int(np.argmax(proba))
        runner_idx = int(np.argsort(proba)[-2])

        predicted_moa = self.label_mapping[pred_idx]
        runner_moa = self.label_mapping[runner_idx]
        confidence = float(proba[pred_idx])

        # ================= SHAP =================
        pool = Pool(df_query)
        shap_all = self.model.get_feature_importance(
            data=pool,
            type="ShapValues"
        )[0]

        shap_pred = shap_all[pred_idx][:-1]
        shap_runner = shap_all[runner_idx][:-1]
        shap_diff_full = shap_pred - shap_runner

        # ================= Ligand vs Protein =================
        ligand_features = [
            f for f in self.selected_features
            if not f.startswith("PROT_")
        ]

        protein_features = [
            f for f in self.selected_features
            if f.startswith("PROT_")
        ]

        ligand_idx = [
            self.selected_features.index(f) for f in ligand_features
        ]

        protein_idx = [
            self.selected_features.index(f) for f in protein_features
        ]

        ligand_strength = np.sum(np.abs(shap_diff_full[ligand_idx]))
        protein_strength = np.sum(np.abs(shap_diff_full[protein_idx]))

        total = ligand_strength + protein_strength

        ligand_pct = round(100 * ligand_strength / total, 1) if total > 0 else 0
        protein_pct = round(100 * protein_strength / total, 1) if total > 0 else 0

        driver = (
            "Ligand-driven"
            if ligand_strength > protein_strength
            else "Protein-driven"
        )

       # ================= MULTIMODAL EXPLANATION =================

        ecfp_scores = {}
        maccs_scores = {}
        descriptor_scores = {}

        for f in self.selected_features:

            idx = self.selected_features.index(f)
            shap_diff = shap_pred[idx] - shap_runner[idx]
            score = float(shap_diff)

            # ================= ECFP =================
            if f.startswith("ECFP_"):

                bit = int(f.split("_")[1])

                if bit in bit_info:
                    for atom_idx, radius in bit_info[bit]:

                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                        if not env:
                            continue

                        submol = Chem.PathToSubmol(mol, env)
                        frag = Chem.MolToSmiles(submol)

                        if frag is None:
                            continue

                        if frag not in ecfp_scores or abs(score) > abs(ecfp_scores[frag]):
                            ecfp_scores[frag] = score


            # ================= MACCS (SMARTS-based) =================
            elif f.startswith("MACCS_"):

                bit = int(f.split("_")[1])

                if bit in MACCS_SMARTS:

                    smarts, count_required = MACCS_SMARTS[bit]

                    # Skip undefined keys (1, 125, 166 handled internally by RDKit)
                    if smarts != '?':

                        patt = Chem.MolFromSmarts(smarts)

                        if patt:

                            matches = mol.GetSubstructMatches(patt)

                            # Respect count logic from RDKit
                            if count_required == 0 and len(matches) > 0:
                                pass
                            elif count_required > 0 and len(matches) > count_required:
                                pass
                            else:
                                continue

                            for match in matches:
                                submol = Chem.PathToSubmol(mol, match)
                                frag = Chem.MolToSmiles(submol)

                                if frag not in maccs_scores or abs(score) > abs(maccs_scores[frag]):
                                    maccs_scores[frag] = score

            # ================= DESCRIPTORS =================
            elif f in self.descriptor_cols:

                value = df_query.iloc[0][f]

                descriptor_scores[f] = {
                    "value": round(float(value), 3),
                    "discrimination_score": round(score, 4),
                    "direction": "increases"
                    if score > 0 else "decreases"
                }


        # ---------------- FORMAT RESULTS ----------------

        top_ecfp = [
            {"fragment": k, "discrimination_score": round(v, 4)}
            for k, v in sorted(
                ecfp_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:8]
        ]

        top_maccs = [
            {"fragment": k, "discrimination_score": round(v, 4)}
            for k, v in sorted(
                maccs_scores.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:6]
        ]

        top_descriptors = sorted(
            [
                {
                    "descriptor": k,
                    "value": v["value"],
                    "discrimination_score": v["discrimination_score"],
                    "direction": v["direction"]
                }
                for k, v in descriptor_scores.items()
            ],
            key=lambda x: abs(x["discrimination_score"]),
            reverse=True
        )[:6]


        return {
                "Predicted_MOA": predicted_moa,
                "Runner_Up_MOA": runner_moa,
                "Confidence": round(confidence, 4),

                "Ligand_Percent": ligand_pct,
                "Protein_Percent": protein_pct,
                "Driver": driver,

                "Top_ECFP_Fragments": top_ecfp,
                "Top_MACCS_Fragments": top_maccs,
                "Top_Descriptors": top_descriptors,

                # 👇 Required for template
                "Top_Fragments": top_ecfp
        }