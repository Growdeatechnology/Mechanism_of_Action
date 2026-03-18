from django.shortcuts import render
from django.views import View

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import (
    MOARequestSerializer,
    MOAResponseSerializer,
    MOAQuerySerializer
)

from .models import MOAQuery

from .services.moa_predictor import MOAPredictor
from .services.similarity_engine import SimilarityEngine
from .services.explanation_engine import ExplanationEngine
from .services.signature_engine import SignatureEngine
from .services.shift_analysis_engine import ShiftAnalysisEngine
from .services.uniprot_service import UniProtService
from .services.feature_extractor import extract_protein_embedding
import re


# ============================================================
# 🌐 FRONTEND HOME PAGE (FULL DASHBOARD)
# ============================================================

class HomeView(View):

    def get(self, request):
        return render(request, "index.html")

    def post(self, request):

        # ---------------------------------
        # Extract & sanitize inputs
        # ---------------------------------
        smiles = (request.POST.get("smiles") or "").strip()
        protein_sequence = (request.POST.get("protein_sequence") or "").strip()
        uniprot_id = (request.POST.get("uniprot_id") or "").strip().upper()

        # 🔹 Clean protein sequence formatting
        if protein_sequence:
            protein_sequence = (
                protein_sequence
                .strip()
                .replace("\n", "")
                .replace(" ", "")
            )


        context = {}

        # ---------------------------------
        # Validate UniProt ID format (if provided)
        # ---------------------------------
        if uniprot_id and not re.match(r"^[A-Z0-9]{6,10}$", uniprot_id):
            context["error"] = "Invalid UniProt ID format."
            return render(request, "index.html", context)

        # ---------------------------------
        # If UniProt ID provided → fetch sequence
        # ---------------------------------
        if uniprot_id and not protein_sequence:
            try:
                protein_sequence = UniProtService.fetch_sequence(uniprot_id)
            except Exception as e:
                context["error"] = f"UniProt fetch failed: {str(e)}"
                return render(request, "index.html", context)

        # ---------------------------------
        # Final validation
        # ---------------------------------
        if not smiles:
            context["error"] = "SMILES is required."
            return render(request, "index.html", context)

        if not protein_sequence:
            context["error"] = "Either Protein Sequence or UniProt ID is required."
            return render(request, "index.html", context)

        try:
            # -----------------------------
            # 1️⃣ Prediction
            # -----------------------------
            predictor = MOAPredictor()
            prediction = predictor.predict(smiles, protein_sequence)

            # -----------------------------
            # 2️⃣ Similarity (Multimodal)
            # -----------------------------

            similarity_engine = SimilarityEngine()

            prot_vector = extract_protein_embedding(protein_sequence)

            similarity = similarity_engine.search(
                smiles,
                prot_vector=prot_vector,
                top_k=10
            )

            # -----------------------------
            # 3️⃣ Explanation
            # -----------------------------
            explanation_engine = ExplanationEngine()
            explanation = explanation_engine.explain(smiles, protein_sequence)

            # -----------------------------
            # 4️⃣ Shift Analysis
            # -----------------------------
            shift_engine = ShiftAnalysisEngine()
            shift = shift_engine.analyze(
                smiles=smiles,
                top_k=80,
                bins=6
            )

            # -----------------------------
            # 5️⃣ Signatures
            # -----------------------------
            signature_engine = SignatureEngine()
            signatures = signature_engine.get_signatures()

            # -----------------------------
            # Save query to database
            # -----------------------------
            try:
                MOAQuery.objects.create(
                    smiles=smiles,
                    protein_sequence=protein_sequence,
                    predicted_moa=prediction.get("Predicted_MOA"),
                    confidence=prediction.get("Confidence"),
                    uncertain=prediction.get("Uncertain", False),
                )
            except Exception:
                pass  # Prevent DB issues from breaking UI

            # -----------------------------
            # Send context to template
            # -----------------------------
            context = {
                "prediction": prediction,
                "similarity": similarity,
                "explanation": explanation,
                "shift": shift,
                "signatures": signatures,
                "smiles": smiles,
                "protein_sequence": protein_sequence,
                "uniprot_id": uniprot_id,
            }

        except Exception as e:
            context["error"] = f"Execution failed: {str(e)}"

        return render(request, "index.html", context)


# ============================================================
# 2.1 + 2.6 MOA PREDICTION (API)
# ============================================================

class MOAPredictView(APIView):

    def post(self, request):

        serializer = MOARequestSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        smiles = serializer.validated_data["smiles"]
        protein_sequence = serializer.validated_data["protein_sequence"]

        if protein_sequence:
            protein_sequence = (
                protein_sequence
                .strip()
                .replace("\n", "")
                .replace(" ", "")
            )

        try:
            predictor = MOAPredictor()
            result = predictor.predict(smiles, protein_sequence)
        except Exception as e:
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)

# ============================================================
# 2.2 SIMILARITY SEARCH (API)
# ============================================================

class SimilaritySearchView(APIView):

    def post(self, request):

        smiles = request.data.get("smiles")
        protein_sequence = request.data.get("protein_sequence")

        if protein_sequence:
            protein_sequence = (
                protein_sequence
                .strip()
                .replace("\n", "")
                .replace(" ", "")
            )
        
        try:
            top_k = int(request.data.get("top_k", 10))
        except:
            top_k = 10


        if not smiles:
            return Response(
                {"error": "SMILES required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            engine = SimilarityEngine()

            if protein_sequence:
                prot_vector = extract_protein_embedding(protein_sequence)
            else:
                prot_vector = None

            result = engine.search(
                smiles,
                prot_vector=prot_vector,
                top_k=int(top_k)
            )

        except Exception as e:
            return Response(
                {"error": f"Similarity search failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)


# ============================================================
# 2.3 EXPLANATION ENGINE (API)
# ============================================================

class ExplanationView(APIView):

    def post(self, request):

        serializer = MOARequestSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        smiles = serializer.validated_data["smiles"]
        protein_sequence = serializer.validated_data["protein_sequence"]
   
        if protein_sequence:
            protein_sequence = (
                protein_sequence
                .strip()
                .replace("\n", "")
                .replace(" ", "")
            )
        try:
            engine = ExplanationEngine()
            result = engine.explain(smiles, protein_sequence)
        except Exception as e:
            return Response(
                {"error": f"Explanation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)


# ============================================================
# 2.4 MOA SIGNATURES (API)
# ============================================================

class SignatureView(APIView):

    def get(self, request):

        try:
            engine = SignatureEngine()
            result = engine.get_signatures()
        except Exception as e:
            return Response(
                {"error": f"Signature engine failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)


# ============================================================
# 2.5 SHIFT ANALYSIS (API)
# ============================================================

class ShiftAnalysisView(APIView):

    def post(self, request):

        smiles = request.data.get("smiles")
        top_k = int(request.data.get("top_k", 80))
        bins = int(request.data.get("bins", 6))

        if not smiles:
            return Response(
                {"error": "SMILES required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            engine = ShiftAnalysisEngine()
            result = engine.analyze(
                smiles=smiles,
                top_k=top_k,
                bins=bins
            )
        except Exception as e:
            return Response(
                {"error": f"Shift analysis failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)


# ============================================================
# QUERY HISTORY (API)
# ============================================================

class QueryHistoryView(APIView):

    def get(self, request):

        logs = MOAQuery.objects.all().order_by("-created_at")[:50]
        serializer = MOAQuerySerializer(logs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)