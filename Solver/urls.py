from django.urls import path,include
from . import views

urlpatterns = [
    path("", view = views.solve, name="Solver"),
    path("verify/",view = views.verify, name="Verify"),
    path("save-solution/",view = views.saveSolution, name="SaveSolution"),
    path("save-json/",view = views.saveModifiedJson, name="SaveJSON"),
    path("solution/",view=views.showSolution,name="Solution")
]