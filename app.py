from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/output_images'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = tf.keras.models.load_model('brain_tumor_classification_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tumor_names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'Pituitary',
    3: 'No Tumor'
}

def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_tumor(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = tumor_names.get(predicted_class_index, "Unknown")
    is_tumor_present = predicted_class_name != 'No Tumor'
    return predicted_class_name, is_tumor_present

def draw_bounding_box(image_path, is_tumor_present):
    img = cv2.imread(image_path)
    if is_tumor_present:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', '').strip()
        patient_age = request.form.get('patient_age', '').strip()
        if not patient_name or not patient_age:
            return render_template('index.html', error='Please enter patient name and age.', patient_name='', patient_age='')
        if 'file' not in request.files:
            return render_template('index.html', error='No file part', patient_name='', patient_age='')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file', patient_name='', patient_age='')
        if file:
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            predicted_class_name, is_tumor_present = predict_tumor(upload_path)
            img_with_box = draw_bounding_box(upload_path, is_tumor_present)
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(result_path, img_with_box)
            return render_template(
                'index.html',
                prediction=predicted_class_name,
                result_image=url_for('static', filename=f'output_images/{filename}'),
                patient_name=patient_name,
                patient_age=patient_age
            )
    # GET request
    return render_template('index.html', patient_name='', patient_age='')

if __name__ == '__main__':
    app.run(debug=True) 