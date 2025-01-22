from pymongo import MongoClient
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *

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