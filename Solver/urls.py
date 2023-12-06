from django.urls import path,include
from . import views

urlpatterns = [
    path("", view = views.solve, name="Solver"),
    path("verify/",view = views.verify, name="Verify"),
    path("solve/",view=views.solve1,name="Solving"),
    path("solution/",view=views.showSolution,name="Solution")

]