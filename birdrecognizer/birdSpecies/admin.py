from django.contrib import admin

from .models import Species

from django.forms import TextInput, Textarea
from django.db import models


class SpeciesAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.CharField: {'widget': TextInput(attrs={'size': '20'})},
        models.TextField: {'widget': Textarea(attrs={'rows': 4, 'cols': 40})},
    }


admin.site.register(Species, SpeciesAdmin)
