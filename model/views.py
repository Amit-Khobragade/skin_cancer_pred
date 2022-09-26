from importlib.resources import path
import os
import re
from unittest import result

from .form import Image, ImageForm
from rest_framework import viewsets
from django.core.files.storage import default_storage
from rest_framework.response import Response
from rest_framework import status
from .models import Image
from .serializer import ImageSerializer

from SKP.settings import BASE_DIR, CLASSES, SKP_MODEL
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from django.shortcuts import render
import tensorflow as tf
import numpy as np

# Create your views here.


class ImageView(viewsets.ModelViewSet):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer


def status(img):
    try:
        model = tf.keras.models.load_model(
            '/mnt/d/code/i2i/PYSKP/SKP/model/SkinCancerPredictor.h5')
        prediction = model.predict(img)
        return prediction
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def FormView(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        # if form.is_valid():
        img = request.FILES['image']
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, img)
        file_url = os.path.join(BASE_DIR, 'media/', file_name_2)
        # original = load_img(file_url, target_size=(224, 224))
        original = load_img(file_url, target_size=(28, 28))
        ar = img_to_array(original)
        # ar = ar.reshape(1, 224, 224, 3)
        ar = ar.reshape(1, 28, 28, 3)

        predictions = SKP_MODEL.predict(ar)
        print(predictions)

        index = np.argmax(predictions[0])

        return render(request, 'results.html', {'data': str(CLASSES[index])})

    form = ImageForm()
    return render(request, 'form.html', {"form": form})
