from django.urls import path

from . import views

app_name = 'birdspecies'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:bird_id>/', views.detail, name='detail')
]
