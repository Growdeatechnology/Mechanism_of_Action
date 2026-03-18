from django.apps import AppConfig


class MoaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "moa"

    def ready(self):
        """
        This runs once when Django starts.
        We trigger model + engine initialization here.
        """

        from .services import model_loader
        from .services import similarity_engine
        from .services import signature_engine
        from .services import shift_analysis_engine

        print("\n🚀 MOA Intelligence Engine Initialization Complete")