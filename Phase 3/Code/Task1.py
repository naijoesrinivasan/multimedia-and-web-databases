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

k = st.number_input('Enter k for Latent Semantics',placeholder="Type a number...",format = "%d",min_value=1,max_value=4338)

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():
            ft = feature_collection.find_one({'_id':0})
            sim = similarity_collection.find_one({'_id':0})   
            for key in feature_collection.find_one({'_id':0}).keys():
                print("KEY: "+str(key)+" TYPE: "+str(type(ft[key])))
            for key in sim.keys():
                print("KEY: "+str(key)+" TYPE: "+str(type(sim[key])))
            ls_even_by_label(k,feature_collection, odd_feature_collection, similarity_collection,caltech101)
else:
    st.write("")