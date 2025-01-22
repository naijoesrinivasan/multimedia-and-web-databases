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
from pathlib import Path
from sys import argv

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)
dbName = 'CSE515-MWD-Vaishnavi-ProjectPhase2'

if len(argv) > 1:
	if argv[1] == "even":
		collection = connect_to_db(dbName,'image_features')
		push_even_to_mongodb(caltech101,collection)	
	
	else:
		collection = connect_to_db(dbName,'image_features_odd')
		push_odd_to_mongodb(caltech101,collection)
    
else:
    print('Mention Odd or Even and run it again')
