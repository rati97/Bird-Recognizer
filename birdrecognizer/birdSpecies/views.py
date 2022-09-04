from django.shortcuts import render, get_object_or_404

from .models import Species


def index(request):
    bird_species = Species.objects.order_by('name')
    context = {
        'bird_species': bird_species
    }
    return render(request, 'birdSpecies/index.html', context)


def detail(request, bird_id):
    bird = get_object_or_404(Species, pk=bird_id)
    return render(request, 'birdSpecies/detail.html', {'bird': bird})
