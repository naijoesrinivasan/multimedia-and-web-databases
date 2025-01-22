import numpy as np
import streamlit as st

from MongoDB.MongoDBUtils import *


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
mod_path = Path(__file__).parent.parent
caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

feature_model = st.selectbox(
        "Select Feature Space",
        ("Color Moments", "Histograms of Oriented Gradients(HOG)", "ResNet-AvgPool-1024","ResNet-Layer3-1024","ResNet-FC-1000","RESNET"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

k = st.number_input('Enter K for Dim Reduction',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

dimred = st.selectbox(
        "Select Dimensionality Reduction Technique",
        ("SVD", "NNMF", "LDA","k-Means"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

if st.button("Run", type="primary"):

    with st.spinner('Calculating...'):
        with st.container():        
            ls3(feature_model, dimred, k, odd_feature_collection, feature_collection, similarity_collection, caltech101)