# Hand Gesture Recognition App
![img.png](img.png)

The Hand Gesture Recognition App is developed in Python using several libraries such as TensorFlow, Keras, MediaPipe, OpenCV, and Tkinter. This application is designed to recognize hand gestures in real-time through a webcam feed. Gestures are classified using a trained Convolutional Neural Network (CNN) Model.

The application trains a model for the gesture classification or loads a pre-trained model if available. It utilizes the VGG16 architecture as a base model, with additional Dense and Dropout layers. During training, data augmentation techniques are applied to the training data to improve the model's performance and ability to generalize. Once the model is trained or loaded, it starts predicting the gestures appearing in the webcam feed.

The application's graphical user interface (GUI) is built with Tkinter, where it displays the webcam feed and the prediction result. 

## Dependencies

- Python
- TensorFlow
- Keras
- MediaPipe
- OpenCV
- Tkinter
- Numpy
- Matplotlib
- PIL
- scikit-learn

## Dataset

This application leverages a hand gesture dataset tailored towards comprehension of the American Sign Language (ASL) letters "A", "F", "L", and "Y". The dataset is available on [Kaggle](https://www.kaggle.com/datasets/joelbaptista/hand-gestures-for-human-robot-interaction/) and was compiled by Joel Baptista.

The dataset incorporates multiple user inputs to establish a robust training backbone for the algorithm. The efficacy of the application, in real-world scenarios, is notably enhanced due to the varied and comprehensive nature of the data.

The ensemble consists of approximately 30,000 images which are bifurcated into a "train" set, deployed for model training, and a "multi_user_test" set, used for evaluation purposes.

## Usage

If you want to use the application, follow these steps:

1. Clone the repository
2. Install necessary dependencies from `requirements.txt` using pip:

    ```bash
    pip install -r requirements.txt
    ```
3. Run `main.py` to start the application:

    ```bash
    python main.py
    ```

## License

This application is part of a postgraduate project. The original dataset used for this project is licensed under the original authors' terms. It was obtained freely from Kaggle and was created by Joel Baptista.

**NOTE:** This application is intended for educational purposes. The developers are not responsible for any misuse or any damages caused by this application. 

## Disclaimer

This application is not endorsed by or affiliated with Joel Baptista or Kaggle. The information provided here is for educational purposes only.

Application was developed by: Kamil Wesołowski, Bartosz Orliński and Bogdan Wójcik.