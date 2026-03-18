from django.db import models


class MOAQuery(models.Model):
    """
    Stores a single MOA prediction request.
    """

    # ================= INPUT =================
    smiles = models.TextField()
    protein_sequence = models.TextField()

    # ================= PREDICTION =================
    predicted_moa = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    uncertain = models.BooleanField(default=False)

    full_probabilities = models.JSONField(null=True, blank=True)

    # ================= EXPLANATION =================
    explanation = models.JSONField(null=True, blank=True)

    # ================= SHIFT ANALYSIS =================
    shift_analysis = models.JSONField(null=True, blank=True)

    # ================= METADATA =================
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_moa} ({round(self.confidence or 0, 3)})"