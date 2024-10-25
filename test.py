import streamlit as st

st.title("Webcam Test")

image = st.camera_input("Capture Image")

if image:
    st.image(image)