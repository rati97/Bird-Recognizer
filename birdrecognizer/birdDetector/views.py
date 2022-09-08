from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadImageForm
import cv2
import numpy as np
from . import Bird_Recognition
from io import BytesIO
import base64

detector = Bird_Recognition.birdRecognizer(classifier_path="birdDetector/400_bird_species_EFFNetB0",
                                           object_detector_path="birdDetector/Efficientdet_d2",
                                           class_names_path="birdDetector/class_names.txt")


def index(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = np.asarray(
                bytearray(request.FILES['image'].read()), dtype="uint8")
            image = cv2.imdecode(
                image, cv2.IMREAD_COLOR)
            result = detector.detect_bird_species(image)
            _, buffer = cv2.imencode(".jpg", result)
            output = BytesIO(buffer)
            im_data = output.getvalue()
            data_url = 'data:image/jpg;base64,' + \
                base64.b64encode(im_data).decode()
            request.session['result_url'] = data_url
            return HttpResponseRedirect('success/')
    else:
        form = UploadImageForm()

    return render(request, 'birdDetector/index.html', {'form': form})


def success(request):
    result_url = request.session.get('result_url', None)
    return render(request, 'birdDetector/success.html',
                  {
                      'result': result_url
                  })
