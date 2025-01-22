#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:53:34 2023

@author: nikhilvr
"""

import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
from torch.autograd import Variable
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
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import preprocessing as p
import pickle
import scipy.io
from tensorly.decomposition import parafac
import tensorly as tl
import scipy.misc
import tensortools as tt
from tensortools.operations import unfold as tt_unfold, khatri_rao
from tensorly import unfold as tl_unfold
import os
import networkx as nx
from sklearn.preprocessing import MinMaxScaler


from FeatureDescriptors.SimilarityScoreUtils import *
from Utilities.DisplayUtils import *
import streamlit as st
from pathlib import Path

from MongoDB.MongoDBUtils import *

from multiprocessing import Process
from multiprocessing import set_start_method
set_start_method("fork")


mod_path = Path(__file__).parent.parent
mat_file_path = str(mod_path)+"/Phase2/LatentSemantics/"


mat = scipy.io.loadmat(mat_file_path+'arrays.mat')

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

personalization_dict = {}

# Create a graph
G = nx.DiGraph()

image_to_label = []

for i in range(0,len(caltech101)):
    image_to_label.append("")




def scale_scores_to_list(similarities):
    
    scaled_scores = {}
    scores = list(similarities.values())
    
    scores = MinMaxScaler().fit_transform(np.array(scores).reshape(-1,1))
    
    index =0
    for key in similarities.keys():
        scaled_scores[key] = scores[index]
        index=index+1
        
    
    #print('SCALED SCORES')
    #print(scaled_scores)
    return scaled_scores



def create_nodes(feature_collection,startIndex,endIndex):
   for i in range(startIndex,endIndex,2):
       image_data = feature_collection.find_one({"_id": i})
       G.add_node(int(i))
       image_to_label[i] = image_data['label']
       #G.add_node(i)
       personalization_dict[i] = 0
       print(i)
    



def append_similarity_graph(index,feature,similarities, n,l):

    if feature == 'color_moments':
        similar = dict(sorted(similarities["color_moments"].items(), key = lambda x: x[1]))
    elif feature == 'hog_descriptor':
        similar = dict(sorted(similarities["hog_descriptor"].items(), key = lambda x: x[1] , reverse=True))
    elif feature == 'avgpool_descriptor':
        similar = dict(sorted(similarities["avgpool_descriptor"].items(), key = lambda x: x[1]), reverse=True)
    elif feature == 'layer3_descriptor':
        similar = dict(sorted(similarities["layer3_descriptor"].items(), key = lambda x: x[1]))
    elif feature == 'fc_descriptor':
         similar = scale_scores_to_list(dict(sorted(similarities["fc_descriptor"].items(), key = lambda x: x[1]), reverse=True))
    
    count=0
    
    sim_scores = []

    for i in range(0,len(caltech101),2):
        if(count>n):
            break
        
        if(int(i)%2 == 0):
            G.add_edge(index, int(i))
            #G[index][int(i)]['weight'] = similar[str(i)]
            if(similar[str(i)] < 0):
                print('Negative')
            sim_scores.append(similar[str(i)])
            count=count+1
        
    personalization_dict[int(index)] = np.mean(sim_scores)
    return G

    


if __name__=="__main__": 

    final_similarity_scores = {}

    dbName = "CSE515-MWD-ProjectPhase2-Final"
    odd_feature_collection = connect_to_db(dbName,'image_features_odd')
    feature_collection = connect_to_db(dbName,'image_features')
    similarity_collection = connect_to_db(dbName,'image_similarities')
        
        #def main():
    # User inputs
    n = int(input("Enter the value of n (number of similar images to consider): "))
    m = int(input("Enter the value of m (number of significant images to identify): "))
    l = int(input("Enter the label (l): "))
    fd = str(input("Enter feature descriptor: "))
   
    
    print('Adding nodes')
    
    processes = [Process(target = create_nodes,args = (feature_collection,idx,idx+250)) for idx in range(0,8677,250)]
    
    print("Threads Running")
    for p in processes:

        p.start()

    

    for p in processes:

        p.join()
    
    print("Threads Done")
    
    print(G)
    
    # Fetch similarities and append graph (Assuming you have already computed the similarities)
    for index in range(0,len(caltech101),2):
        similarities = similarity_collection.find_one({"_id": index})
        #image_data = feature_collection.find_one({"_id": index})
        #print('Scores')
        #print(similarities[fd])
        #print(similarities)
        #print(type(similarities))
        append_similarity_graph(index,fd,similarities, n,l)
    
    print('Graph Generated complete')
        
    print(G)
    
    print('Personalization Dict Complete')
    
    #print(personalization_dict)
    
    personalized_pagerank = nx.pagerank(G, alpha=0.85,personalization=personalization_dict)
    
    print(personalized_pagerank)
    
    required_images = []
    for key in personalized_pagerank.keys():
        if(image_to_label[key] == l):
            data=[]
            data.append(key)
            data.append(personalized_pagerank[key])
            required_images.append(data)
            
            
    required_images = sorted(required_images,key=lambda x : x[1],reverse = True)[:m]
    
    print('The top M images for the given label are ')
    
    print(required_images)
            
            
        
    
    
   
    
    
    


   



