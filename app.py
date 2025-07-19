import streamlit as st # for creating app interface
import tensorflow as tf
import numpy as np
from PIL import Image # for image processing

# function to load the pre trained model and cache it for optimized performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('flower_model_trained.hdf5')
    return model

# Function to predict the class of the input image using the loaded model
def predict_class(image, model):
    image = tf.cast(image, tf.float32) # convert image to float32
    image = tf.image.resize(image, [180, 180]) # resize image to match the model input shape
    image = np.expand_dims(image, axis=0) # add extra dimension to match the model input shape
    predictions = model.predict(image) # make predictions using the model
    return predictions # returning predicted class probabilities 

model = load_model() # load the model
st.title("Flower Classification") # set the title of the app

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"]) # create a file uploader to upload images

if file is None:
    st.text("Waiting for upload...")
    # Displaying a message if no file is uploaded yet

else:
    slot = st.empty()
    slot.text("Running inference...") # Displaying a message while the model is running
    
    test_image = Image.open(file) # opening the uploaded image

    st.image(test_image, caption="Input Image", width=400) # displaying the uploaded image

    pred = predict_class(np.asarray(test_image), model) # predicting the class of the uploaded image

    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'] # list of class names

    result = class_names[np.argmax(pred)] # getting the class name with the highest probability

    output = "The image is of a " + result # creating the output message

    slot.text("Done!") # Displaying a message when the model is done running

    st.success(output) # displaying the output message as a success notification

    