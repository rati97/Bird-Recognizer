from django.db import models


class Species(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=300, null=True, blank=True)
    image_path = models.ImageField(
        upload_to="bird_images/", null=True, blank=True)
    add_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
