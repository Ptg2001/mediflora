import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define class names and labels
class_names = ['Ajwain', 'Aloevera', 'Amla', 'Anjeer', 'Apta', 'Arali', 'Arka', 'Ashoka', 'Ashwagandha', 'Asthma_weed',
               'Badipala', 'Balloon_Vine', 'Bamboo', 'Basale', 'Basil_seeds', 'Beans', 'Betel', 'Black_nightshade',
               'Brahmi', 'Bringaraja', 'camphor', 'Caricature', 'Castor', 'Chakte', 'Chilly', 'Citron_lime',
               'Coffee', 'Common_rue', 'Coral_jasmine', 'Coriender', 'Crape_jasmine', 'Curry', 'Custard_Apple',
               'Drumstick', 'Fenugreek', 'German_turnip', 'Ginger', 'Globe_Amaranth', 'Grape_leaf',
               'Green_chiretta', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Insulin', 'Jackfruit', 'Jamaica', 'Jamun',
               'Jasmine', 'Jungle_geranium', 'Kanagale', 'Karanda', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango',
               'Marigold', 'Mehndi', 'Mexican_Mint', 'Mint', 'Monsoon_leaves', 'Neem', 'Nerale', 'Nilgiri', 'Nooni', 'Oleander',
               'Onion', 'Padri', 'Papaya', 'Pea', 'Peepal', 'Pepper', 'Pomegranate', 'Poppy_seeds', 'Pumpkin', 'Raddish',
               'Raimuniya', 'Rajgira', 'Rasna', 'Rose', 'Rose_Apple', 'Sadabahar', 'Sampige', 'Sandalwood',
               'Sapota', 'Sita_ashok', 'Spinach', 'Tamarind', 'Taro', 'Tecoma', 'Thumbai', 'Tomato', 'Tulsi', 'Turmeric']

class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

# Function to preprocess an image
def pre_process(img_path, target_size=(150, 150)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image

# Function to load and preprocess the dataset
def load_data(data_dir, target_size=(224, 224), split_ratio=0.8):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)

        print(f"Loading images from {class_dir}")

        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)

            try:
                image = pre_process(img_path, target_size)

                if np.random.rand() < split_ratio:
                    train_images.append(image)
                    train_labels.append(class_names_label[class_name])
                else:
                    test_images.append(image)
                    test_labels.append(class_names_label[class_name])
            except (cv2.error, Exception) as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

    if not train_images:
        print("No training images loaded. Check the data directory and format.")
    if not test_images:
        print("No testing images loaded. Check the data directory and format.")

    train_images = np.array(train_images, dtype='float32') / 255.0
    train_labels = np.array(train_labels, dtype='int32')
    test_images = np.array(test_images, dtype='float32') / 255.0
    test_labels = np.array(test_labels, dtype='int32')

    return (train_images, train_labels), (test_images, test_labels)

# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = load_data('Medicinal', target_size=(224, 224))

# Data Augmentation and Image Preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create your custom classifier
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(nb_classes, activation='softmax')(x)

# Create the full model by combining the ResNet50 base and the custom classifier
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Define a learning rate schedule
initial_learning_rate = 0.0001
def lr_schedule(epoch):
    if epoch < 5:
        return initial_learning_rate
    else:
        return initial_learning_rate * tf.math.exp(-0.1 * (epoch - 5))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Create data generators with data augmentation
batch_size = 32
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

# Train the model with data augmentation and the learning rate schedule
history = model.fit(train_generator, epochs=20, validation_data=test_generator,
                    callbacks=[early_stopping, lr_scheduler])

# After training, count the number of trained and non-trained images
total_trained_images = sum(1 for label in train_labels if label < nb_classes)
total_non_trained_images = len(train_labels) - total_trained_images

# Print the counts
print(f"Total Trained Images: {total_trained_images}")
print(f"Total Non-Trained Images: {total_non_trained_images}")

# Save the trained model
model.save("resnet50_model.h5")

# Save the training history
with open("training_history_resnet50.pkl", "wb") as history_file:
    pickle.dump(history.history, history_file)

# Plot accuracy and loss
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
