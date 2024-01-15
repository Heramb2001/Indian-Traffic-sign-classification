import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your data directories
train_dir = r'C:\Users\bhher\Desktop\indian_traffic\indian\TCS\Train'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# Load the trained model
model = load_model("modell.h5")

class_labels = list(train_generator.class_indices.keys())

st.markdown("<h2 style='color: #007BFF;'>Indian Traffic Sign Classification</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    #Display krde bhai image
    st.image(img[0])

    # Display krde predicted wale ko 
    st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: red;'>{class_labels[predicted_class]}</p>", unsafe_allow_html=True)
    
    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions.tolist(), f)

    st.success("Predictions saved successfully!")
