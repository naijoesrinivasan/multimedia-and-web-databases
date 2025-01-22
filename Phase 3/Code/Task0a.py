from torchvision.datasets import Caltech101
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *
from Task0.Utils import get_inherent_dim_even

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():    
            features, feature_name  = get_inherent_dim_even()
            st.write(feature_name)
            st.write(features.shape)
            st.write(features)
else:
    st.write("")