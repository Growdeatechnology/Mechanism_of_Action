from django.urls import path
from django.http import JsonResponse
from . import views


# ------------------------------------------------------------
# ROOT API STATUS
# ------------------------------------------------------------
def api_home(request):
    return JsonResponse({
        "status": "MOA Intelligence Engine Running",
        "version": "1.0",
        "available_endpoints": {
            "POST /predict/": "MOA prediction",
            "POST /similarity/": "Closest known ligand search",
            "POST /explain/": "Model explanation",
            "GET /signatures/": "MOA chemical signatures",
            "POST /shift-analysis/": "Trend-based MOA shift analysis",
            "GET /history/": "Recent query history"
        }
    })


urlpatterns = [

    # Root endpoint
    path("", views.HomeView.as_view(), name="home"),

    # Core MOA Prediction
    path("predict/", views.MOAPredictView.as_view(), name="moa-predict"),

    # Similarity Search
    path("similarity/", views.SimilaritySearchView.as_view(), name="moa-similarity"),

    # Explanation
    path("explain/", views.ExplanationView.as_view(), name="moa-explain"),

    # Chemical Signatures
    path("signatures/", views.SignatureView.as_view(), name="moa-signatures"),

    # Shift Analysis
    path("shift-analysis/", views.ShiftAnalysisView.as_view(), name="moa-shift-analysis"),

    # Query History
    path("history/", views.QueryHistoryView.as_view(), name="moa-history"),
]