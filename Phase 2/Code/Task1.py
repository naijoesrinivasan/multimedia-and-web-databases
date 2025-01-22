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
from tqdm.notebook import tqdm
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)
dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')


label = st.number_input('Enter Image Label',placeholder="0",format = "%d",min_value=0,max_value=100)
k = st.number_input('Enter k for similar images',placeholder="0",format = "%d",min_value=1,max_value=8676)
feature_space = st.selectbox(
        "Select Feature Space",
        ("Color Moments", "Histograms of Oriented Gradients(HOG)", "ResNet-AvgPool-1024","ResNet-Layer3-1024","ResNet-FC-1000"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )


if st.button("Run", type="primary"):
	with st.spinner('Calculating...'):
		with st.container():
    		similarity_calculator_by_label(label, feature_space, k,odd_feature_collection, feature_collection, similarity_collection, caltech101)