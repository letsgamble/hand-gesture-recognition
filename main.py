import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disabling the usage of a physical GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Setting TensorFlow log level.
# 1: INFO, 2: WARNING, 3: ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GestureRecognitionApp:
    """
    A class used to contain the Hand Gesture Recognition Application.

    The class contains methods to train or load a pre-trained model, create a model,
    judicially schedule learning rates, plot learning curves, calculate bounding
    rectangles, isolate hands in frames, and predict gestures.
    """

    def __init__(self):
        """
        Initializes the object with initial configurations, a TKinter window, and classes.
        Also, creates an instance of the hands class from the MediaPipe library.
        """
        self.image_size = (224, 224, 3)
        self.batch_size = 32
        self.dataset_dir = 'HGR dataset'
        self.model_file = 'gesture_model.h5'

        self.train_generator, self.validation_generator = self.setup_generators()
        self.index_to_gesture = {v: k for k, v in self.train_generator.class_indices.items()}
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition")
        self.cap = cv2.VideoCapture(0)
        self.lmain = tk.Label(self.root)
        self.lmain.pack()
        self.prediction_label = tk.Label(self.root, text="Prediction")
        self.prediction_label.pack()
        self.hands = mp.solutions.hands.Hands()
        self.hand_feed_label = tk.Label(self.root)
        self.hand_feed_label.pack()

    def setup_generators(self):
        """
        Sets up data generators for training and validation subsets of the dataset.
        The generators apply data augmentation (rotation, zoom, shift, shear, flip).

        Returns:
        Tuple: containing train_data and validation_data generator objects.
        """
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
        )

        train_generator = datagen.flow_from_directory(
            self.dataset_dir + '/train/',
            target_size=(self.image_size[0], self.image_size[1]),
            batch_size=self.batch_size, class_mode='categorical')
        validation_generator = datagen.flow_from_directory(
            self.dataset_dir + '/multi_user_test/',
            target_size=(self.image_size[0], self.image_size[1]),
            batch_size=self.batch_size, class_mode='categorical')

        return train_generator, validation_generator

    def train_or_load_model(self):
        """
        Trains the model from scratch if a pre-trained model isn't available.
        Loads the model if it's already available. Also compiles and prints the
        model summary.
        """
        model_path = os.path.abspath('gesture_model.keras')
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
        lr_scheduler_callback = LearningRateScheduler(self.lr_scheduler)

        if os.path.exists(model_path):
            print("Loading existing model.")
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        else:
            # If model doesn't exist, begin training process.
            print("Model not found. Starting training process.")
            self.model = self.create_model()

            early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                           restore_best_weights=True,
                                           verbose=1)
            model_checkpoint = ModelCheckpoint(model_path, save_best_only=True,
                                               verbose=1)

            self.history = self.model.fit(
                self.train_generator,
                validation_data=self.validation_generator,
                epochs=30,
                callbacks=[early_stopping, model_checkpoint,
                           lr_scheduler_callback, tensorboard_callback],
                steps_per_epoch=self.train_generator.samples // self.batch_size,
                validation_steps=self.validation_generator.samples // self.batch_size
            )
            self.plot_learning_curves(self.history)

            # Save the trained model.
            self.model.save(model_path)
            print("Model training completed and saved.")
            self.evaluate_model()

    def evaluate_model(self):
        """
        Evaluate the model on the validation dataset and print the classification
        report and confusion matrix.
        """
        predictions = self.model.predict(self.validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.validation_generator.classes

        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes))

        print("Confusion Matrix:")
        print(confusion_matrix(true_classes, predicted_classes))

    def create_model(self):
        """
        Creates a Sequential model with pre-trained VGG16 as base model.
        Compiled with 'Adam' optimizer and 'Categorical Crossentropy' as loss function.

        Returns:
        The created and compiled model.
        """
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.image_size)

        for layer in base_model.layers[:]:
            layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def lr_scheduler(self, epoch, lr):
        """
        Learning rate scheduling function that decreases the learning rate as epochs increase.

        Parameters:
        epoch (int): The current epoch number passed automatically by LearningRateScheduler.
        lr (float): The learning rate of the previous epoch passed automatically by LearningRateScheduler.

        Returns:
        new_lr (float): The new calculated learning rate.
        """
        # Adjust learning rate depending on the epoch
        new_lr = 0.001 * 0.1 ** (epoch / 30)
        print(f"Epoch {epoch + 1}: Adjusting learning rate to {new_lr:.6f}.")
        return new_lr

    @staticmethod
    def plot_learning_curves(history):
        """
        Plots the Training vs Validation Accuracy and Loss curves.

        Parameters:
        history (tensorflow.python.keras.callbacks.History): Object returned by model.fit() function.
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc='upper right')
        plt.show()

    def predict_gesture(self, hand_img):
        """
        Predicts the hand gesture shown in the image.

        Parameters:
        hand_img (numpy.ndarray): An RGB image of a hand.

        Returns:
        gesture_name (str): The name of the predicted gesture.
        confidence (float): The confidence score of the prediction.
        hand_img_resized (numpy.ndarray): The resized version of original hand image.
        """
        h, w = hand_img.shape[:2]
        scale = min(self.image_size[0] / h, self.image_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(hand_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad the resized image if it's not the target size
        delta_w = max(self.image_size[0] - new_w, 0)
        delta_h = max(self.image_size[1] - new_h, 0)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        hand_img_resized = cv2.copyMakeBorder(resized, top, bottom, left,
                                              right, cv2.BORDER_CONSTANT,
                                              value=[0, 0, 0])
        # Normalize and reshape for model prediction
        hand_img_normalized = hand_img_resized / 255.0
        hand_img_reshaped = np.expand_dims(hand_img_normalized,
                                           axis=0)

        # Model prediction
        predictions = self.model.predict(hand_img_reshaped, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        gesture_name = self.index_to_gesture[predicted_class]
        if hand_img_resized.shape != self.image_size:
            raise ValueError(
                f"Processed image shape {hand_img_resized.shape} does not match expected shape {self.image_size}")

        return gesture_name, confidence, hand_img_resized

    def isolate_hand(self, frame):
        """
        Identifies the hand region in a frame and crops the region.

        Parameters:
        frame (numpy.ndarray): An RGB image from which hand needs to be isolated.

        Returns:
        Tuple (frame, img): frame with identified hand region,
                            isolated hand from the frame if found, else None.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                x, y, w, h = self.calculate_bounding_rect(hand_landmarks)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                return frame, frame[y:y + h, x:x + w]
        return frame, None

    def calculate_bounding_rect(self, landmarks):
        """
        Calculates the bounding rectangle for the identified hand region.

        Parameters:
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):

        Returns:
        Bounding rectangle coordinates.
        """
        frame_width, frame_height = self.cap.get(3), self.cap.get(4)
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]

        x_min = min(x_coords) * frame_width
        x_max = max(x_coords) * frame_width
        y_min = min(y_coords) * frame_height
        y_max = max(y_coords) * frame_height

        # Adding a margin to the bounding box
        margin_x = (x_max - x_min) * 0.5  # 10% margin
        margin_y = (y_max - y_min) * 0.5  # 10% margin

        x_min = max(0, x_min - margin_x)
        x_max = min(frame_width, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(frame_height, y_max + margin_y)

        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def update(self):
        """
        Updates the application window with frames captured from the webcam feed
        and prediction results of the model.
        """
        ret, frame = self.cap.read()
        if ret:
            marked_frame, hand_img = self.isolate_hand(frame)
            display_text = "No hand detected"
            if hand_img is not None and hand_img.size > 0:
                gesture_name, confidence, processed_hand_img = self.predict_gesture(
                    hand_img)
                display_text = f"{gesture_name} ({confidence:.2f}%)"

                # Update hand feed label with the resized and processed hand image
                hand_img_rgb = cv2.cvtColor(processed_hand_img,
                                            cv2.COLOR_BGR2RGB)
                hand_img_tk = ImageTk.PhotoImage(
                    image=Image.fromarray(hand_img_rgb))
                self.hand_feed_label.imgtk = hand_img_tk
                self.hand_feed_label.configure(image=hand_img_tk)

            # Update the main camera feed display
            cv_img = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            self.prediction_label.config(text=display_text)

        self.lmain.after(10, self.update)

    def run(self):
        """
        Executes the application.
        """
        if not self.model:
            print("Model not found. Please train the model first.")
            return
        self.update()
        self.root.mainloop()


if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.train_or_load_model()
    app.run()
