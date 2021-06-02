import streamlit as st
import tensorflow as tf

classes = ['Ripe Brinjal','Ripe Mango','Ripe Tomoto','Unripe Brinjal','Unripe Mango','Unripe Tomoto']

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    new_fruit_model = tf.keras.models.load_model(
    "model.h5",
    custom_objects=None,
    compile=True
    )
    return new_fruit_model


    #print("Loaded model from disk")
    return model
model = load_model()
st.write("""
            Brinjal, Mango and Tomoto Classifier
         """
        )
file=st.file_uploader("Please upload an image of Brinjal or Mango or Tomoto",type=["jpg"])
#import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):

    size=(150,150)
    image= ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    result = model.predict(img_reshape)
    result1 = result[0]
    for i in range(6):
        if result1[i] == 1.:
            break;
    prediction = classes[i]
    return prediction
    


if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    #class_names=['Mango','Jackfruit']
    string="This image is most likely:"+predictions
    st.success(string)
