from django.urls import path
from . import views

urlpatterns = [
    path('dbshow/', views.dbshowFunc, name='dbshow'),
]
