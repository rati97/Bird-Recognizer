from django.urls import path


from . import views

app_name = 'birdspecies'
urlpatterns = [
    path('', views.BirdListView.as_view(
        template_name="birdSpecies/index.html"), name='index'),
    path('<int:bird_id>/', views.detail, name='detail')
]
