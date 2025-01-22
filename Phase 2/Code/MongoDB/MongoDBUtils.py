import pymongo
from pymongo import MongoClient
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

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *

import streamlit as st
from pathlib import Path

#Connect to Database and retrieve collection

def connect_to_db(dbname, collectionName):
    client = MongoClient('localhost',27017)
    db = client[dbname]
    collection = db[collectionName]
    print('Connected to '+str(collection))
    return collection

def get_client():
    client = MongoClient('localhost',27017)
    return client

def get_collection(client,dbname,collectionName):
    db = client[dbname]
    collection = db[collectionName]
    print('Connected to '+str(collection))
    return collection

def close_client(client):
    client.close()

"""def push_dataset_to_mongodb(dataset,collection):
    for idx in tqdm(range(len(dataset))):
        image = cv2.cvtColor(np.array(dataset.__getitem__(index = idx)[0]),cv2.COLOR_RGB2BGR)
        descriptors = descriptor_calculator(image, idx,dataset)
        collection.update_one({'_id':idx},{'$set':descriptors},upsert = True)"""

def push_even_to_mongodb(dataset,collection):
    for idx in tqdm(range(0,len(dataset),2)):
        image = cv2.cvtColor(np.array(dataset.__getitem__(index = idx)[0]),cv2.COLOR_RGB2BGR)
        descriptors = descriptor_calculator(image, idx,dataset)
        collection.update_one({'_id':idx},{'$set':descriptors},upsert = True)

def push_odd_to_mongodb(dataset,collection):
    for idx in tqdm(range(1,len(dataset),2)):
        image = cv2.cvtColor(np.array(dataset.__getitem__(index = idx)[0]),cv2.COLOR_RGB2BGR)
        descriptors = descriptor_calculator(image, idx,dataset)
        collection.update_one({'_id':idx},{'$set':descriptors},upsert = True)