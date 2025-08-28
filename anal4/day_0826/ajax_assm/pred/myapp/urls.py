from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict_salary_ajax/', views.predict_salary_ajax, name='predict_salary_ajax'),
]
