import streamlit as st
import requests
import os
import base64

# frontend title
st.title('Image Upload for YOLOv8 Object Detection')

# upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# check if the file exists
if uploaded_file is not None:
    # turn binary file into base64 string
    strfile = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    # send request
    response = requests.post("http://localhost:8000/upload", 
                             json={"files": strfile,
                                   "name":uploaded_file.name})
    # check if the request was successful
    if response.status_code == 200:
        # combine the file path
        pic_path = os.path.normpath(os.path.join(response.json()['img_path'], uploaded_file.name))
        # display the processed image
        st.image(f"{pic_path}", caption='Processed Image')
        # display a success message
        st.write("Image processed successfully.")
    else:
        # display an error message
        st.write("Failed to process image.")
