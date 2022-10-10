from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np
import cv2

class Sentiment():
    def __init__(self,emotion_model_path,face_model_path):
        """
        We instantiate the Sentiment class with the pretrained model paths
        Args:
            - emotion_model_path (str): path to the keras emotion detection model
            - face_model_path (str): path to the haar cascade opencv model
        """
        #Create a sequential keras model: 
        #https://keras.io/api/models/sequential/
        model = Sequential()
        
        #We will add different layers according to the pretrained architecture
        #https://keras.io/api/layers/
        #The input to the model are grayscale 48x48 images.
        #The first convolutional block should have 2 convolutional layers followed by a max pooling and dropout layers
        #The first convolutional layer should have 32 3x3 filters with relu activation. 
        #The second convolutional layer should have 64 3x3 filters with relu activation.
        #The max pooling layer should have a 2x2 window.
        #The dropout layer should have a 0.25 dropout rate
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        #The next convolutional block should have:
        #Convolutional layer (128 3x3 filters with relu) 
        #Max pooling layer with 2x2 window
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #The next convolutional block has the same structure as the previous one:
        #Convolutional layer (128 3x3 filters with relu) 
        #Max pooling layer with 2x2 window
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #The final block of the model has:
        #Dropout layer with 0.25 dropout rate
        #Flatten layer
        #Dense layer with 1024 neurons and relu activation
        #Dropout layer with 0.5 dropout rate
        #Final Dense layer with 7 neurons with softmax activation
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        #We already created the model, we only need to load the weights. 
        #https://www.tensorflow.org/tutorials/keras/save_and_load
        model.load_weights(emotion_model_path)
        
        #Save the model as a internal class variable
        self.emotion_model=model
        
        #The emotion dictionary maps the output of the network to the name of the emotion.
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        #The face detection model is already implemented in opencv:
        #https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
        #Pretrained models: https://github.com/opencv/opencv/tree/3.4/data/haarcascades
        self.face_model = cv2.CascadeClassifier(face_model_path)

    def predict(self,frame):
        """
        The predict method should process the image and return the transformed image with the faces and emotions detected.
        Args:
            - frame (np.array): The image as a numpy array with shape (W,H,3)
        Returns:
            - (np.array): Transformed image 
        """
        #The model was trained on grayscale images, therefore we must convert the color image (W,H,3) into grayscale (W,H,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #The face model returns a list of bounding boxes (x,y,w,h) for every detected face
        faces = self.face_model.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            #We draw a blue rectangle corresponding to the bounding box
            #https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            
            #We use slicing to extract the portion of the image inside the bounding box 
            roi_gray = gray[y:y + h, x:x + w]
            
            #We resize the image to a size of (48,48) which is the input of the model
            resized_roi = cv2.resize(roi_gray, (48, 48))
            
            #We need to add a batch dimension and a channel dimension
            #The input to the model should have shape (1,48,48,1)
            cropped_img = np.expand_dims(np.expand_dims(resized_roi, -1), 0)
            
            #We run the emotion detection model and get the softmax output
            prediction = self.emotion_model.predict(cropped_img)
            
            #We get the name of the emotion from the model's output
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]
            
            #We add the emotion text to the bounding box
            #https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
            cv2.putText(frame, emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return frame