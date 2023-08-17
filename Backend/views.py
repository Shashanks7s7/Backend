import os
from PIL import Image
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage

labelling = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Poliferative DR'
}

def home(request):
    context = {
        'url': '',
        'result': ''
    }
    if request.method == 'POST':
        image = request.FILES.get('image')
        print(image)
        if image:
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            prediction = process_image(image)
            prediction_list = prediction.tolist()
            max_value, max_index = find_highest_number_with_index(
                prediction_list[0])
            context.update({
                'url': filename,
                'result': labelling.get(max_index),
                'prediction': prediction_list[0]
            })
            return render(request, "index.html", context)
    return render(request, "index.html", context)


def process_image(image):
    img = Image.open(image)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    model_path = os.path.join(os.path.dirname(
        __file__), 'Models', 'baseline1.h5')
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(img)
    print(predictions)
    return predictions


def find_highest_number_with_index(lst):
    if not lst:
        return None, None
    max_value = lst[0]
    max_index = 0
    for index, value in enumerate(lst):
        if value > max_value:
            max_value = value
            max_index = index
    return max_value, max_index
