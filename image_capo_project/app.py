from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']= 1
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods= ['GET','POST'])
def after():
    file = request.files['file1']
    file.save("D:/image_capo_project/file.jpg")
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug= True)
import streamlit as st
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

captioning_model = load_model('model.h5')

st.title("image cap gen")

uploaded_file = st.file_uploader("choose file", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption= "uploaded image", use_column_width=True)


    def generate_caption(image):
        img=load_img(image, target_size=(299,299))
        img=img_to_array(img)
        img=preprocess_input(img)
        img= np.expand_dims(img, axis=0)


        caption = predict_caption(img)
        return caption
    
    def predict_caption(image):
        caption = captioning_model.predict(image)
        return caption
    
    if st.button("generate caption"):
        if captioning_model is not None:
            generated_caption = generate_caption(uploaded_file)
            st.write("generated caption:", generated_caption)

        else:
            st.write("please load a model first")