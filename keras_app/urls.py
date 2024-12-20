from django.urls import path
from . import views

urlpatterns = [
  path('', views.IndexView.as_view(), name='index'),
  path('chap04', views.Chap04View.as_view(), name='chap04'),
  path('chap05', views.Chap05View.as_view(), name='chap05'),
  path('chap06', views.Chap06View.as_view(), name='chap06'),
  path('chap08', views.Chap08View.as_view(), name='chap08'),
]