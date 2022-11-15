from turtle import st
from tempfile import NamedTemporaryFile

import PIL.Image
import io

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.models import *
from src.constant import Path


@st.cache(allow_output_mutation=True)
def load_model(path: str ) :
    """Retrieves the trained model"""
    model = Model(model_path)
    return model

@st.cache()
def predict(
    img,
    model,
    k):
    formatted_out = model.predict_proba(img,k)
    return formatted_out

if __name__ == '__main__':

    st.title(" Image Classification Project ")

    st.write(
        " Goals of this project to classify image into 6 categories: ",
        "\n Building, Forest, Glacier, Mountain, Sea, Street. "
        )

    #Load the model
    mod ={
        "MobileNet-V2" : Path.MODEL_MOBILE , 
        "Efficientnet-B0" : Path.MODEL_B0, 
        "Efficientnet-B7" : Path.MODEL_B7
    }

    model_type = st.sidebar.selectbox(
        "Select model", list(mod.keys()))
    
    model_path = mod[model_type]
       
    model = load_model(model_path)

    st.write("You could upload an Image below or select from the box on the left side :wink:")
    #Option for uploading teh photo by the user
    file = st.file_uploader('Upload An Image')

    st.set_option('deprecation.showfileUploaderEncoding', False)
    temp_file = NamedTemporaryFile(delete=False)

    if file:  # if user uploaded file
        temp_file.write(file.getvalue())        
        img = PIL.Image.open(temp_file.name)

        prediction = predict(img, model, k=5)  
        
    else:
        dat = {
                "Sample Images " : Path.PRED_PATH
        }

        dataset_type = st.sidebar.selectbox(
            "Data ", list(dat.keys()))

        image_files = dat[dataset_type]

        avail_img = os.listdir(image_files)

        image_name = st.sidebar.selectbox(
            "Image", avail_img
        )

        pth = os.path.join(image_files,image_name)

        img = PIL.Image.open(pth)
        
        prediction = predict(img, model, k=5)  


    #Show Image
   
    st.header("Here is the image you selected")
    rsz = img.resize((224,224))
    st.image(rsz)

    st.header("Here are the top prediction")

    #Create temp df
    df = pd.DataFrame(data = np.zeros((5,2)),
                        columns = ['Label','Percentage'],
                        index=np.arange(1,6)
                        )

    if prediction :
        for ix, pred in enumerate(prediction):
            df.iloc[ix,0] = pred[0]
            df.iloc[ix,1] = pred[1]

    # st.write(df.to_html(escape=False), unsafe_allow_html=True)

    st.caption(f" :small_blue_diamond: The result is **{df['Label'][1]}** with probability **{df.iloc[0,1]}**")
    st.table(df)
