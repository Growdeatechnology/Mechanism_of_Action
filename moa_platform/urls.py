from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # MOA Web UI
    path('', include('moa.urls')),

    # Optional future API namespace
    # path('api/', include('moa.api_urls')),
]

# Static & Media (development only)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)