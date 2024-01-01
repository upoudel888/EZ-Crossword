from django.urls import path,include
from . import views

urlpatterns = [
    path("", view = views.generate, name="Generator"),
    path('result/',view = views.showGeneratedCrossword,name="generation_result"),
    path('download-json/', view = views.download_json, name='download_json'),
]