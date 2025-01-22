from torchvision.datasets import Caltech101
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

option = st.selectbox(
        "Select Classifier",
        ("Decision Tree", "PPR","Nearest Neighbors"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

if option == "Nearest Neighbors":
    k = st.number_input('Enter k for k-Means classifier',placeholder="Type a number...",format = "%d",min_value=1,max_value=4338)
    img_id = st.number_input('Enter ImageID for query',placeholder="Type a number...",format = "%d",min_value=1,max_value=8677)
    teleport_prob = 0
elif option=="PPR":
    k = 0
    img_id = st.number_input('Enter ImageID for query',placeholder="Type a number...",format = "%d",min_value=1,max_value=8677)
    teleport_prob = st.number_input('Enter teleportation probability for PageRank',placeholder="Type a number...",format="%f",min_value=0.0,max_value=1.0)
else:
    k = 0
    img_id = st.number_input('Enter ImageID for query',placeholder="Type a number...",format = "%d",min_value=1,max_value=8677)
    teleport_prob = 0

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():    
            classifier(img_id,option,feature_collection,odd_feature_collection,similarity_collection,caltech101,k=k,teleport_prob = teleport_prob)
else:
    st.write("")