from django.urls import path,include
from . import views

urlpatterns = [
    path("", view = views.generate, name="Generator"),
    path('verify/',view = views.verifyGeneratedGrid,name="verify_result"),
    path('result/',view = views.showGeneratedCrossword,name="show_result"),
    path('save-solution/',view = views.saveSolution,name="save_result"),
    path('download-json/', view = views.download_json, name='download_json'),
]