import cv2
import numpy as np
import torch
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
import statistics


import streamlit as st
from pathlib import Path

from Utilities.DisplayUtils import *


#Color Moments
def featurenormalize(feature_vector):
    # Normalize the feature vector to have unit length
    if np.isnan(feature_vector).any():
        # Handle the presence of NaN values (e.g., replace with zeros)
        feature_vector = np.nan_to_num(feature_vector)

    # Calculate the L2 norm of the feature_vector
    norm = np.linalg.norm(feature_vector)
    return feature_vector / norm if norm != 0 else feature_vector

def euclidean_distance_calculator(x, y):
    return np.linalg.norm(x - y)

def similarity_score_color_moments(moments1, moments2):
    # Normalize the feature vectors
    moments1_normalized = featurenormalize(moments1)
    moments2_normalized = featurenormalize(moments2)
    
    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(moments1_normalized, moments2_normalized)

    return round(similarity,5)



#Similarity for HOG

def cosine_similarity_calculator(descriptor1, descriptor2, normalize=True):
    if normalize:
        descriptor1 = featurenormalize(descriptor1)
        descriptor2 = featurenormalize(descriptor2)
        
    # Reshape descriptors as needed
    descriptor1 = np.reshape(descriptor1, (1, -1))
    descriptor2 = np.reshape(descriptor2, (1, -1))

    similarity = cosine_similarity(descriptor1, descriptor2)
    return similarity[0][0]

def similarity_score_hog(descriptor1, descriptor2):
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)

#Cosine Similarity for Avgpool
def similarity_score_avgpool(descriptor1, descriptor2):
    # Calculate Cosine Similarity for avgpool descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)


#Layer3

def similarity_score_layer3(descriptor1, descriptor2):
    # Calculate Euclidean distance between two layer3 descriptors
    desc1_normalized = featurenormalize(descriptor1)
    desc2_normalized = featurenormalize(descriptor2)

    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(desc1_normalized, desc2_normalized)

    return round(similarity,5)

#FC

def similarity_score_fc(descriptor1, descriptor2):
    # Convert to NumPy arrays if they are of type torch.Tensor
    if isinstance(descriptor1, torch.Tensor):
        descriptor1 = descriptor1.numpy()
    if isinstance(descriptor2, torch.Tensor):
        descriptor2 = descriptor2.numpy()

    # Calculate Cosine Similarity for fc descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]), 5)

def get_ksimilar_labels(imagedata1,feature_collection,dataset,feature_space):
    #dict to store the similarity scores according to the label
    sim_la = {}
    for cmpidx in tqdm(range(0,len(dataset),2)):
        imagedata2 = feature_collection.find_one({"_id": cmpidx})
        label = imagedata2["label"]
        #Calculating the similarity scores according to the given feature space.
        if feature_space == "Color Moments":
            similarity = similarity_score_color_moments(imagedata1["color_moments"], imagedata2["color_moments"])
        elif feature_space == "Histograms of Oriented Gradients(HOG)":
            similarity = similarity_score_hog(imagedata1["hog_descriptor"], imagedata2["hog_descriptor"])
        elif feature_space == "ResNet-AvgPool-1024":
            similarity = similarity_score_avgpool(imagedata1["avgpool_descriptor"], imagedata2["avgpool_descriptor"])
        elif feature_space == "ResNet-Layer3-1024":
            similarity = similarity_score_layer3(imagedata1["layer3_descriptor"], imagedata2["layer3_descriptor"]) 
        elif feature_space == "ResNet-FC-1000":
            similarity = similarity_score_fc(imagedata1["fc_descriptor"], imagedata2["fc_descriptor"])
        elif feature_space == "RESNET":
            similarity = get_similarity_score_resnet(imagedata1["fc_softmax_descriptor"], imagedata2["fc_softmax_descriptor"])

        #Storing similarity scores
        if label in sim_la:
            sim_la[label].append(similarity)
        else: 
            sim_la[label] = []
            sim_la[label].append(similarity)

    return sim_la

def get_k_similar_labels_old(idx,similarity_collection,feature_collection,dataset,feature_space):
    sim_la = {}
    print("Inside get)k_similar_labels_old\n")
    f_spaces = similarity_collection.find_one({'_id': idx})
    for cmpidx in tqdm(range(0,len(dataset),2)):
        imagedata2 = feature_collection.find_one({"_id": cmpidx})
        label = imagedata2["label"]
        #Retrieving the similarity scores according to the given feature space.
        if feature_space == "Color Moments":
            similarity = f_spaces["color_moments"][str(cmpidx)]
        elif feature_space =="Histograms of Oriented Gradients(HOG)":
            similarity = f_spaces["hog_descriptor"][str(cmpidx)]
        elif feature_space ==  "ResNet-AvgPool-1024":
            similarity = f_spaces["avgpool_descriptor"][str(cmpidx)]
        elif feature_space == "ResNet-Layer3-1024":
            similarity = f_spaces["layer3_descriptor"][str(cmpidx)]
        elif feature_space == "ResNet-FC-1000":
            similarity = f_spaces["fc_descriptor"][str(cmpidx)]
        elif feature_space == "RESNET":
            similarity = f_spaces["fc_softmax_descriptor"][str(cmpidx)]
        #Storing similarity scores
        if label in sim_la:
            sim_la[label].append(similarity)
        else: 
            sim_la[label] = []
            sim_la[label].append(similarity)

    return sim_la

def similarity_calculator(index,odd_feature_collection,feature_collection,similarity_collection,dataset):

    similarities = similarity_collection.find_one({'_id': index})
    if similarities!=None:
        # print("Similarity score present for "+str(index)+" returning from DB")
        return similarities

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
            "fc_descriptor": {},
            "fc_softmax_descriptor": {}
        }
    
    for cmpidx in tqdm(range(0,len(dataset),2)):
        
        imagedata2 = feature_collection.find_one({"_id": cmpidx})

        color_moments_similarity = similarity_score_color_moments(imagedata1["color_moments"], imagedata2["color_moments"])
        histogram_similarity = similarity_score_hog(imagedata1["hog_descriptor"], imagedata2["hog_descriptor"])
        avgpool_similarity = similarity_score_avgpool(imagedata1["avgpool_descriptor"], imagedata2["avgpool_descriptor"])
        layer3_similarity = similarity_score_layer3(imagedata1["layer3_descriptor"], imagedata2["layer3_descriptor"]) 
        fc_similarity = similarity_score_fc(imagedata1["fc_descriptor"], imagedata2["fc_descriptor"])
        fc_sofmax_similarity = get_similarity_score_resnet(imagedata1["fc_softmax_descriptor"], imagedata2["fc_softmax_descriptor"])

        if not np.isnan(color_moments_similarity):
            similarities["color_moments"][str(cmpidx)] = color_moments_similarity
        else: 
            similarities["color_moments"][str(cmpidx)] = 1
        similarities["hog_descriptor"][str(cmpidx)] =  histogram_similarity
        similarities["avgpool_descriptor"][str(cmpidx)] =  avgpool_similarity
        similarities["layer3_descriptor"][str(cmpidx)] = layer3_similarity
        similarities["fc_descriptor"][str(cmpidx)] =  fc_similarity
        similarities["fc_softmax_descriptor"][str(cmpidx)] =  fc_sofmax_similarity
    
    similarity_collection.update_one({'_id':index},{'$set':similarities},upsert = True)
    
    return similarities

def similarity_calculator_by_label(label,feature_space,k,odd_feature_collection,feature_collection,similarity_collection,dataset):
    
    print("Entry similarity_calculator_by_label for: "+str(label))
    image_data_by_label = feature_collection.find({'label':label})
    final_scores = []
    
    required_indices_for_label = []
    
    #Segregate images of the label in particular, and calculate similarity scores
    for doc in image_data_by_label:
        required_indices_for_label.append(doc['_id'])
        similarity_calculator(doc['_id'], odd_feature_collection, feature_collection, similarity_collection, dataset)
        
        
    #print('The required indices for this label are')
    #print(required_indices_for_label)

    if feature_space == "Color Moments":
        feature_space = "color_moments"

    elif feature_space == "Histograms of Oriented Gradients(HOG)":
        feature_space = "hog_descriptor"

    elif feature_space == "ResNet-AvgPool-1024":
        feature_space = "avgpool_descriptor"

    elif feature_space == "ResNet-Layer3-1024":
        feature_space = "layer3_descriptor"

    elif feature_space == "ResNet-FC-1000":
        feature_space = "fc_descriptor"

    elif feature_space == "RESNET":
        feature_space = "fc_softmax_descriptor"
        
    
   #Extract required scores and calculate average
    for index in required_indices_for_label:
        required_scores = []
        avg_score=0.0
        image_similarity_scores = similarity_collection.find_one({'_id': index})
        #print(image_similarity_scores[feature_space].keys())
        for dct_idx in required_indices_for_label:
            required_scores.append(image_similarity_scores[feature_space][str(dct_idx)])
       # print(required_scores)
        required_scores =[1 if x>1 else x for x in required_scores]
        #print(required_scores)
        avg_score=statistics.mean(required_scores)
        imagedata = feature_collection.find_one({'_id': index})
        final_data={'imageId':imagedata['_id'],'image':cv2.cvtColor(np.array(imagedata['image'], dtype=np.uint8), cv2.COLOR_BGR2RGB),'average_score':avg_score}
        final_scores.append(final_data)
    
    #To test the output stream
    #st.write(final_scores)
    
    #Sort the result dict to retrieve top k results
    final_scores = sorted(final_scores, key = lambda x: x['average_score'],reverse=True)[:k]
    
    #Format final output to call display method
    
    display_images_list=[]
    display_indices=[]
    display_similarity_scores=[]
    return_idx = int(final_scores[0]['imageId'])
    
    """for score in final_scores:
        display_images_list.append(score['image'])
        display_indices.append(score['imageId'])
        display_similarity_scores.append(score['average_score'])"""
        
    #print("The lengths are :")
    #print(str(len(display_images_list))+" "+str(len(display_indices))+" "+str(len(display_similarity_scores)))
        
    #Call display method for final output
    
    #display_images(display_images_list,display_indices,display_similarity_scores,0,0,"Similarity Score : ")
    print("Exit similarity_calculator_by_label")
    return return_idx
           
def similarity_calculator_newimg(imagedata1,odd_feature_collection,feature_collection,similarity_collection,dataset):

    similarities = {
            "color_moments": {},
            "hog_descriptor": {},
            "avgpool_descriptor": {},
            "layer3_descriptor": {},
            "fc_descriptor": {},
            "fc_softmax_descriptor": {}
        }
    
    for cmpidx in tqdm(range(0,len(dataset),2)):
        
        imagedata2 = feature_collection.find_one({"_id": cmpidx})

        color_moments_similarity = similarity_score_color_moments(imagedata1["color_moments"], imagedata2["color_moments"])
        histogram_similarity = similarity_score_hog(imagedata1["hog_descriptor"], imagedata2["hog_descriptor"])
        avgpool_similarity = similarity_score_avgpool(imagedata1["avgpool_descriptor"], imagedata2["avgpool_descriptor"])
        layer3_similarity = similarity_score_layer3(imagedata1["layer3_descriptor"], imagedata2["layer3_descriptor"]) 
        fc_similarity = similarity_score_fc(imagedata1["fc_descriptor"], imagedata2["fc_descriptor"])
        fc_sofmax_similarity = get_similarity_score_resnet(imagedata1["fc_softmax_descriptor"], imagedata2["fc_softmax_descriptor"])
        if not np.isnan(color_moments_similarity):
            similarities["color_moments"][str(cmpidx)] = color_moments_similarity
        else: 
            similarities["color_moments"][str(cmpidx)] = 1
        similarities["hog_descriptor"][str(cmpidx)] =  histogram_similarity
        similarities["avgpool_descriptor"][str(cmpidx)] =  avgpool_similarity
        similarities["layer3_descriptor"][str(cmpidx)] = layer3_similarity
        similarities["fc_descriptor"][str(cmpidx)] =  fc_similarity
        similarities["fc_softmax_descriptor"][str(cmpidx)] =  fc_sofmax_similarity
    
    return similarities


def get_similarity_score_resnet(query_image_features,image_features):
  
    dot_product = np.dot(query_image_features, image_features)
    norm1 = np.linalg.norm(query_image_features)
    norm2 = np.linalg.norm(image_features)
    similarity = dot_product / (norm1 * norm2)
    
    return similarity