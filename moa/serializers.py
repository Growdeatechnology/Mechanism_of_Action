from rest_framework import serializers
from .models import MOAQuery


# ============================================================
# INPUT SERIALIZER
# ============================================================

class MOARequestSerializer(serializers.Serializer):
    """
    Input payload for MOA prediction.
    """
    smiles = serializers.CharField()
    protein_sequence = serializers.CharField()

    def validate_smiles(self, value):
        if not value.strip():
            raise serializers.ValidationError("SMILES cannot be empty.")
        return value

    def validate_protein_sequence(self, value):
        if not value.strip():
            raise serializers.ValidationError("Protein sequence cannot be empty.")
        return value


# ============================================================
# PREDICTION OUTPUT SERIALIZER
# ============================================================

class MOAResponseSerializer(serializers.Serializer):
    """
    Output structure returned to frontend.
    """

    Predicted_MOA = serializers.CharField()
    Confidence = serializers.FloatField()
    Uncertain = serializers.BooleanField()

    Full_Probabilities = serializers.DictField()

    Explanation = serializers.DictField(required=False)
    Shift_Analysis = serializers.DictField(required=False)


# ============================================================
# DATABASE SERIALIZER (History / Logging)
# ============================================================

class MOAQuerySerializer(serializers.ModelSerializer):
    """
    Used if you want to retrieve saved queries.
    """

    class Meta:
        model = MOAQuery
        fields = "__all__"