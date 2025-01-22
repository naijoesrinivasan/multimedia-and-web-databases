from numba import njit 
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
import pickle
import json
from MongoDB.MongoDBUtils import *
from multiprocessing import Process
from multiprocessing import set_start_method
set_start_method("fork")

  
#@njit(nopython = False) 
def featurenormalize(feature_vector):
    # Normalize the feature vector to have unit length
    if np.isnan(feature_vector).any():
        # Handle the presence of NaN values (e.g., replace with zeros)
        feature_vector = np.nan_to_num(feature_vector)

    # Calculate the L2 norm of the feature_vector
    norm = np.linalg.norm(feature_vector)
    return feature_vector / norm if norm != 0 else feature_vector

#@njit(nopython = False) 
def euclidean_distance_calculator(x, y):
    return np.linalg.norm(x - y)

#@njit(nopython = False) 
def similarity_score_color_moments(moments1, moments2):
    # Normalize the feature vectors
    moments1_normalized = featurenormalize(moments1)
    moments2_normalized = featurenormalize(moments2)
    
    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(moments1_normalized, moments2_normalized)

    return round(similarity,5)

#Similarity for HOG
#@njit(nopython = False) 
def cosine_similarity_calculator(descriptor1, descriptor2, normalize=True):
    if normalize:
        descriptor1 = featurenormalize(descriptor1)
        descriptor2 = featurenormalize(descriptor2)
    
    # Reshape descriptors as needed
    descriptor1 = np.reshape(descriptor1, (1, -1))
    descriptor2 = np.reshape(descriptor2, (1, -1))

    similarity = cosine_similarity(descriptor1, descriptor2)
    return similarity[0][0]

#@njit(nopython = False) 
def similarity_score_hog(descriptor1, descriptor2):
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)

#Cosine Similarity for Avgpool
#@njit(nopython = False) 
def similarity_score_avgpool(descriptor1, descriptor2):
    # Calculate Cosine Similarity for avgpool descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)

#Layer3
#@njit(nopython = False) 
def similarity_score_layer3(descriptor1, descriptor2):
    # Calculate Euclidean distance between two layer3 descriptors
    desc1_normalized = featurenormalize(descriptor1)
    desc2_normalized = featurenormalize(descriptor2)

    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(desc1_normalized, desc2_normalized)

    return round(similarity,5)

#FC
#@njit(nopython = False) 
def similarity_score_fc(descriptor1, descriptor2):
    # Convert to NumPy arrays if they are of type torch.Tensor
    if isinstance(descriptor1, torch.Tensor):
        descriptor1 = descriptor1.numpy()
    if isinstance(descriptor2, torch.Tensor):
        descriptor2 = descriptor2.numpy()

    # Calculate Cosine Similarity for fc descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]), 5)

#@njit(nopython = False) 
def similarity_calculator(index,odd_feature_collection,feature_collection,similarity_collection,dataset):

    similarities = similarity_collection.find_one({'_id': index})
    if similarities!=None:
        print("Similarity score present for "+str(index)+" returning from DB")
        return similarities,True

    print("Similarity score not present for "+str(index)+" calculating and caching to DB")


    if index%2 == 0:
        imagedata1 = feature_collection.find_one({'_id': index})
    else:
        imagedata1 = odd_feature_collection.find_one({'_id': index})
        
    similarities = {
            "_id": index,
            "color_moments": {},
            "hog_descriptor": {},
            "avgpool_descriptor": {},
            "layer3_descriptor": {},
            "fc_descriptor": {}
        }
    
    for cmpidx in tqdm(range(0,len(dataset),2)):
        
        imagedata2 = feature_collection.find_one({"_id": cmpidx})

        color_moments_similarity = similarity_score_color_moments(imagedata1["color_moments"], imagedata2["color_moments"])
        histogram_similarity = similarity_score_hog(imagedata1["hog_descriptor"], imagedata2["hog_descriptor"])
        avgpool_similarity = similarity_score_avgpool(imagedata1["avgpool_descriptor"], imagedata2["avgpool_descriptor"])
        layer3_similarity = similarity_score_layer3(imagedata1["layer3_descriptor"], imagedata2["layer3_descriptor"]) 
        fc_similarity = similarity_score_fc(imagedata1["fc_descriptor"], imagedata2["fc_descriptor"])
        if not np.isnan(color_moments_similarity):
            similarities["color_moments"][str(cmpidx)] = color_moments_similarity
        else: 
            similarities["color_moments"][str(cmpidx)] = 1
        similarities["hog_descriptor"][str(cmpidx)] =  histogram_similarity
        similarities["avgpool_descriptor"][str(cmpidx)] =  avgpool_similarity
        similarities["layer3_descriptor"][str(cmpidx)] = layer3_similarity
        similarities["fc_descriptor"][str(cmpidx)] =  fc_similarity
    
    similarity_collection.update_one({'_id':index},{'$set':similarities},upsert = True)
    print("Image: "+str(index))
    
    return similarities,False

#@njit(nopython = False) 
def call_similarity_calculator(startindex,endindex,odd_feature_collection,feature_collection,similarity_collection,dataset):

    counter = 0
    for idx in range(endindex,startindex,-2):
        sim_score, fromdb = similarity_calculator(idx,odd_feature_collection,feature_collection,similarity_collection,caltech101)
        if fromdb == True:
            counter+=1
            print(str(counter)+" from db")

if __name__=="__main__": 
    mod_path = Path(__file__).parent.parent
    caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

    final_similarity_scores = {}

    dbName = "CSE515-MWD-Vaishnavi-ProjectPhase2"
    odd_feature_collection = connect_to_db(dbName,'image_features_odd')
    feature_collection = connect_to_db(dbName,'image_features')
    similarity_collection = connect_to_db(dbName,'image_similarities')

    transferclient = MongoClient('192.168.0.5',27017)
    transferdb = transferclient["CSE515-MWD-Nikhil_V_Ramanan-ProjectPhase2"]
    transfercollection = transferdb['image_similarities']


    processes = [Process(target = call_similarity_calculator,args = (idx,idx+500,odd_feature_collection,feature_collection,transfercollection,caltech101)) for idx in range(0,8677,500)]

    print("Threads Running")
    for p in processes:

        p.start()

    print("Threads Done")

    for p in processes:

        p.join()

    #for idx in tqdm(range(0,len(caltech101))):
    #    sim_score = similarity_calculator(idx,odd_feature_collection,feature_collection,similarity_collection,caltech101)