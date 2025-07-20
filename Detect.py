import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('brain_tumor_classification_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tumor_names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'Pituitary',
    3: 'No Tumor'
}

def classify_tumor(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = tumor_names.get(predicted_class_index, "Unknown")

    is_tumor_present = predicted_class_name != 'No Tumor'
    bounding_box_image = draw_bounding_box(image_path, is_tumor_present)
    
    output_image_path = os.path.join('static', 'output_images', os.path.basename(image_path))
    cv2.imwrite(output_image_path, bounding_box_image)

    return predicted_class_name, output_image_path

def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

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

class TumorClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Classifier")
        self.root.geometry("600x700")

        tk.Label(root, text="Brain Tumor Classifier", font=("Arial", 20)).pack(pady=20)
        tk.Button(root, text="Upload Image", command=self.upload_image).pack(pady=10)
        self.canvas = tk.Canvas(root, width=400, height=400, bg="lightgray")
        self.canvas.pack(pady=10)
        tk.Button(root, text="Predict Tumor", command=self.predict_image).pack(pady=10)
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((400, 400))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def predict_image(self):
        if hasattr(self, 'image_path'):
            predicted_class_name, is_tumor_present = predict_tumor(self.image_path)
            result_text = f"Tumor Detected: {predicted_class_name}" if is_tumor_present else "No Tumor Detected"
            self.result_label.config(text=result_text)

            processed_img = draw_bounding_box(self.image_path, is_tumor_present)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            img_with_box = Image.fromarray(processed_img)
            img_with_box = img_with_box.resize((400, 400))
            self.img_tk_with_box = ImageTk.PhotoImage(img_with_box)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk_with_box)

        else:
            messagebox.showwarning("No Image", "Please upload an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorClassifierApp(root)
    root.mainloop()
