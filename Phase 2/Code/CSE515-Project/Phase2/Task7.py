import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.stats import moment
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cosine
from tqdm import tqdm
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

def query_ksimilar_new_label(image):
    color_moments = color_moments_calculator(image)
    hog_descriptor = hog_calculator(image)
    avgpool_descriptor = avgpool_calculator(image)
    layer3_descriptor = layer3_calculator(image)
    fc_descriptor = fc_calculator(image)

    imagedata = {
        'image': image.tolist() if isinstance(image, np.ndarray) else image,  # Convert the image to a list for storage
        'color_moments': color_moments.tolist() if isinstance(color_moments, np.ndarray) else color_moments,
        'hog_descriptor': hog_descriptor.tolist() if isinstance(hog_descriptor, np.ndarray) else hog_descriptor,
        'avgpool_descriptor': avgpool_descriptor.tolist() if isinstance(avgpool_descriptor, np.ndarray) else avgpool_descriptor,
        'layer3_descriptor': layer3_descriptor.tolist() if isinstance(layer3_descriptor, np.ndarray) else layer3_descriptor,
        'fc_descriptor': fc_descriptor.tolist()
    }
    return imagedata

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)
k = st.number_input('Enter k for similar images',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

latsem = st.selectbox(
    "Select the latent semantics",
    ("LS1","LS2","LS3","LS4"),
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    )

feature_model = st.selectbox(
        "Select Feature Space",
        ("Color Moments", "Histograms of Oriented Gradients(HOG)", "ResNet-AvgPool-1024","ResNet-Layer3-1024","ResNet-FC-1000", "RESNET"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

latentk = st.number_input('Enter k for Latent Semantics',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

if latsem!="LS2":
    dimred = st.selectbox(
            "Select Dimensionality Reduction Technique",
            ("SVD", "NNMF", "LDA","k-Means"),
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
else:
    dimred = ""
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpeg', 'jpg'])

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():    
        	get_simlar_ls(idx, feature_model, k,latsem, latentk, dimred, odd_feature_collection, feature_collection, similarity_collection, caltech101)    	
elif st.button("Run for uploaded image", type="primary") and uploaded_file is not None:
    with st.spinner('Calculating...'):
        with st.container():
            #preprocessing the given image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            st.markdown("Query Image")
            
            image = cv2.cvtColor(opencv_image,cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA)
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(image, channels="BGR")
            #Getting image data by processing the image
            imagedata = query_ksimilar_new_label(image)
            get_simlar_ls_img(imagedata, feature_model, k, latsem, latentk, dimred, feature_collection)     
else:
    st.write("")