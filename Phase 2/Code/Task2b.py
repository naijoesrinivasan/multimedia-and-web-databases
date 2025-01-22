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
from heapq import nlargest

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

# Fuction the process the image to get the image data
def get_similar_resnet(query_image_features, feature_collection, dataset):
    sim_la = {}

    for cmpidx in tqdm(range(0,len(dataset),2)):
          
        imagedata2 = feature_collection.find_one({"_id": cmpidx})
        image = np.array(imagedata2['image'], dtype=np.uint8)
        label = imagedata2["label"]
        image_features = fc_calculator_2(image)
        similarity = get_similarity_score_resnet(query_image_features, image_features)

        if label in sim_la:
            sim_la[label].append(similarity)
        else: 
            sim_la[label] = []
            sim_la[label].append(similarity)

    return sim_la

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)
k = st.number_input('Enter k for similar images',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpeg', 'jpg'])

if st.button("Run", type="primary") and uploaded_file is None:
    with st.spinner('Calculating...'):
        avg = {}
        with st.container():
            #Getting image data from the dataset  
            if idx%2 == 0:
                imagedata1 = feature_collection.find_one({'_id': idx})
            else:
                imagedata1 = odd_feature_collection.find_one({'_id': idx})

            query_image = np.array(imagedata1['image'], dtype=np.uint8)
            query_image_features = fc_calculator_2(query_image)

            #display query image
            st.markdown("Query Image\n")
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(query_image, channels="BGR") 
            st.write("Query image label: ", get_class_name(imagedata1["label"]))
            
            similarity = get_similar_resnet(query_image_features, feature_collection, caltech101)

            #Calculating the average for every label
            for i in similarity:
                avg[i] = sum(similarity[i])/len(similarity[i])

            srt = dict(sorted(avg.items(), key = lambda x: x[1], reverse=True)[:k])
            
            #print top k matching labels
            for key, val in srt.items():
                st.write(get_class_name(key), ": ", val)

            st.write("")

elif st.button("Run for uploaded image", type="primary") and uploaded_file is not None:
    with st.spinner('Calculating...'):
        avg = {}
        with st.container():
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            st.markdown("Query Image")
            
            image = cv2.cvtColor(opencv_image,cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA)
            left_co, cent_co,last_co = st.columns(3)
            with cent_co:
                st.image(image, channels="BGR")
            
            query_image_features = fc_calculator_2(image)
            
            similarity = get_similar_resnet(query_image_features, feature_collection, caltech101)

            #Calculating the average for every label
            for i in similarity:
                avg[i] = sum(similarity[i])/len(similarity[i])

            srt = dict(sorted(avg.items(), key = lambda x: x[1], reverse=True)[:k])
            
            #print top k matching labels
            for key, val in srt.items():
                st.write(get_class_name(key), ": ", val)
            st.write("")

else:
    st.write("")