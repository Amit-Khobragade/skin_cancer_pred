from dataclasses import field, fields
from email.mime import image
from django import forms
from .models import Image


class ImageForm(forms.Form):
    class Meta:
        model = Image
        fields = '__all__'

    image = forms.ImageField()
