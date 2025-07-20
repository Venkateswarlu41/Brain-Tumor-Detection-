import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        
        return self.__data_generation(batch_image_paths, batch_labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_labels):
        X = np.empty((self.batch_size, *self.img_size, 1), dtype=np.float32)  
        y = np.empty((self.batch_size), dtype=int)

        for i, img_path in enumerate(batch_image_paths):
            img = cv2.imread(img_path, 0) 
            img = cv2.resize(img, self.img_size)  
            X[i,] = img[..., np.newaxis] / 255.0  
            y[i] = batch_labels[i]

        return X, y

def load_data(data_dir):
    images = []
    labels = []
    class_labels = {
        'glioma_tumor': 0,
        'meningioma_tumor': 1,
        'pituitary_tumor': 2,
        'no_tumor': 3
    }

    for tumor_type, label in class_labels.items():
        tumor_dir = os.path.join(data_dir, tumor_type)
        if not os.path.exists(tumor_dir):
            print(f"Warning: The directory {tumor_dir} does not exist.")
            continue

        for img_file in os.listdir(tumor_dir):
            img_path = os.path.join(tumor_dir, img_file)
            images.append(img_path)
            labels.append(label)

    return images, np.array(labels)

def load_unsupervised_data(data_dir):
    images = []
    modalities = ['flair', 'seg', 't1', 't1ce', 't2']
    for root, dirs, _ in os.walk(data_dir):
        for modality in modalities:
            modality_path = os.path.join(root, modality)
            if os.path.exists(modality_path):
                for img_file in os.listdir(modality_path):
                    img_path = os.path.join(modality_path, img_file)
                    images.append(img_path)

    return images


train_data_dir = 'Train'
test_data_dir = 'Test'
unsupervised_data_dir = 'Brain_tumor_dataset'


X_train, y_train = load_data(train_data_dir)
X_test, y_test = load_data(test_data_dir)
X_unsupervised = load_unsupervised_data(unsupervised_data_dir)


def vision_transformer_block(input_shape, num_patches, d_model):
    inputs = layers.Input(shape=input_shape)
    
    patches = layers.Conv2D(d_model, (num_patches, num_patches), strides=num_patches, padding='valid')(inputs)
    patches_flattened = layers.Reshape((patches.shape[1] * patches.shape[2], d_model))(patches)

    transformer_output = layers.MultiHeadAttention(num_heads=8, key_dim=d_model)(patches_flattened, patches_flattened)
    transformer_output = layers.Add()([patches_flattened, transformer_output])
    transformer_output = layers.LayerNormalization()(transformer_output)

    cls_token = layers.GlobalAveragePooling1D()(transformer_output)

    return Model(inputs, cls_token)

def cnn_block(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.GlobalAveragePooling2D()(x)

    return Model(inputs, x)

input_shape = (224, 224, 1)

vit_model = vision_transformer_block(input_shape, num_patches=16, d_model=64)

vit_output = layers.Dense(16 * 16 * 64)(vit_model.output)
vit_output_reshaped = layers.Reshape((16, 16, 64))(vit_output)

cnn_model = cnn_block((16, 16, 64))
cnn_output = cnn_model(vit_output_reshaped)

dense_output = layers.Dense(256, activation='relu')(cnn_output)
output = layers.Dense(4, activation='softmax')(dense_output)

hybrid_model = Model(vit_model.input, output)

hybrid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


hybrid_model.summary()



train_generator = CustomDataGenerator(X_train, y_train, batch_size=16)
test_generator = CustomDataGenerator(X_test, y_test, batch_size=16)

history = hybrid_model.fit(train_generator,
                           validation_data=test_generator,
                           epochs=100)

# Save the model
hybrid_model.save('brain_tumor_classification_model.h5')
print("Model saved as 'brain_tumor_classification_model.h5'")


# CLASSIFICATION RESULTS


test_loss, test_accuracy = hybrid_model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

predicted_classes = np.argmax(hybrid_model.predict(test_generator), axis=1)

cm = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:\n", cm)

cr = classification_report(y_test, predicted_classes, target_names=['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'])
print("Classification Report:\n", cr)


# PLOTTING ACCURACY AND LOSS


plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
