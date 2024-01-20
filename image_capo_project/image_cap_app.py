import streamlit as st
import torch
from torchvision.transforms import transforms 
from PIL import Image

from image_captioning_prj import generate_caption

st.title("img captioning app")

uploaded_img = st.file_uploader("upload an image", type = ["jpg", "png", "jpeg"])

if uploaded_img is not None :
    image = Image.open(uploaded_img)

    st.image(image, caption = "Uploaded Image", use_column_width= True)


    if st.button("Generate Captions"):
        caption = generate_caption(image, model)
        st.write("generated caption:")
        st.write(caption)
