import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "moa_platform.settings"
)

application = get_wsgi_application()

# 🔥 Optional: Preload ML artifacts once at server startup
try:
    from moa.services.model_loader import registry
    print("\n🚀 MOA Model Loaded Successfully in WSGI")
except Exception as e:
    print("\n⚠️ Model preload failed:", str(e))