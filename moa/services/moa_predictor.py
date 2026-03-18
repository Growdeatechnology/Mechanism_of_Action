# ============================================================
# moa_predictor.py
# FULLY TRAINING-ALIGNED MOA PREDICTOR
# Supports:
# - ECFP
# - MACCS
# - ProtBERT
# - RDKit Descriptors
# ============================================================

import numpy as np
import pandas as pd

from .model_loader import registry
from .feature_extractor import extract_features


CONFIDENCE_THRESHOLD = 0.4


class MOAPredictor:

    def __init__(self):
        if registry.model is None:
            raise ValueError("Model artifacts not loaded.")

    # ========================================================
    # BUILD MODEL-ALIGNED FEATURE VECTOR
    # ========================================================
    def _build_feature_vector(self, smiles, protein_sequence):

        # 1️⃣ Extract raw features
        features = extract_features(smiles, protein_sequence)

        ecfp = features["ecfp"]
        maccs = features["maccs"]
        descriptors = features["descriptors"]
        protein_emb = features["protein_embedding"]

        # 2️⃣ Construct full feature dictionary
        feature_dict = {}

        # ---- ECFP ----
        for i in range(len(ecfp)):
            feature_dict[f"ECFP_{i}"] = int(ecfp[i])

        # ---- MACCS ----
        for i in range(len(maccs)):
            feature_dict[f"MACCS_{i}"] = int(maccs[i])

        # ---- PROT ----
        for i in range(len(protein_emb)):
            feature_dict[f"PROT_{i}"] = float(protein_emb[i])

        # ---- DESCRIPTORS ----
        descriptor_names = registry.descriptor_columns
        for name, value in zip(descriptor_names, descriptors):
            feature_dict[name] = float(value)

        # 3️⃣ Create DataFrame aligned EXACTLY to training order
        feature_columns = registry.feature_columns
        df_query_full = pd.DataFrame([feature_dict], columns=feature_columns)

        # 4️⃣ Apply variance selector
        X_var = registry.selector.transform(df_query_full.values)

        # 5️⃣ Align top selected features EXACTLY
        var_feature_names = np.array(feature_columns)[registry.selector.get_support()]
        df_var = pd.DataFrame(X_var, columns=var_feature_names)

        df_selected = df_var[registry.selected_features]

        return df_selected

    # ========================================================
    # MAIN PREDICTION
    # ========================================================
    def predict(self, smiles, protein_sequence):

        if registry.model is None:
            return {"error": "Model not loaded. Add artifacts first."}

        try:
            X = self._build_feature_vector(smiles, protein_sequence)
        except Exception as e:
            return {"error": str(e)}

        # ---- Model prediction ----
        proba = registry.model.predict_proba(X)[0]

        pred_idx = int(np.argmax(proba))
        runner_idx = int(np.argsort(proba)[-2])

        confidence = float(proba[pred_idx])
        label = registry.label_mapping.get(pred_idx, "Unknown")
        runner_label = registry.label_mapping.get(runner_idx, "Unknown")

        uncertain_flag = confidence < CONFIDENCE_THRESHOLD

        final_label = "Uncertain MOA" if uncertain_flag else label

        full_probs = {
            registry.label_mapping.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "Predicted_MOA": final_label,
            "Raw_Prediction": label,
            "Runner_Up": runner_label,
            "Confidence": round(confidence, 4),
            "Uncertain": uncertain_flag,
            "Threshold": CONFIDENCE_THRESHOLD,
            "Full_Probabilities": full_probs
        }