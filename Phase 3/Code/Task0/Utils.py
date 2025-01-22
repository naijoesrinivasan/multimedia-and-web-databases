import numpy as np
import scipy
import os
import pandas as pd
from pathlib import Path
from Task0.distance_utils import *
from tqdm import tqdm
from scipy.io import savemat, loadmat
import warnings
warnings.filterwarnings("ignore")

mod_path = Path(__file__).parent.parent
ROOT_DIR = str(mod_path)
FEATURES = ['cm_features', 'hog_features', 'avgpool_features', 'layer3_features', 'fc_features', 'resnet_features']
data = scipy.io.loadmat(ROOT_DIR+'/Store/arrays.mat')
latent_space_features = dict()

def create_feature_df_even():
    """
        Method to do dimentionality reduction and stress calculation for each feature descriptor.
        For ever feature descriptor, calculate the inherent dimention and do PCA using maximum likelihood estimation.
        Then calculate stress of the original space images and latentn space images. The stress values for every
        featrue descriptor is stored in a dataframe. latent dimentions are also stored in the memory.
        Args:
            None
        Return:
            
    """

    df_stress = pd.DataFrame(columns=["Feature Space", "stress"])
    print("Doing Dim. reduction for ever feature...")

    for feature in tqdm(FEATURES):
        original_space_data = data[feature]

        original_space_data_norm = featurenormalize(original_space_data)
       
        latent_space_data= pca_mle(original_space_data_norm)
        
        stress = calculate_stress(original_space_data_norm, latent_space_data)

        stress_data = {"Feature Space": feature, "stress": stress}
        df_stress = pd.concat([df_stress, pd.DataFrame([stress_data])], ignore_index=True)
        latent_space_features[feature] = latent_space_data

    savemat(os.path.join(ROOT_DIR, "Store/latent_dim.mat"), latent_space_features)
    df_stress.to_csv(os.path.join(ROOT_DIR, "Store/stress.csv"), index = False)
    return df_stress

def get_best_feature_even(df_stress):
    """
        Using the Stress values finding the feature descriptor with the lowest stress value
        int the latent space.
        Args:
            df_stress(pandas.DataFrame): Dataframe consisting of feature descriptors and stress values in the original
                and latent space. 
        return:
            name of the feature descriptor which best represents the data.
    """
    return df_stress[df_stress["stress"] == df_stress["stress"].min()]["Feature Space"].values[0]

def get_inherent_dim_even():
    """
        Method to return the inherent dimentions to return from the list of feature spaces having best stress.
        Args:
            None
        Return:
            Numpy array consisting of the feature representation of even numbered data.   
    """
    df_stress = create_feature_df_even()
    best_feature = get_best_feature_even(df_stress)

    inherent_dim = loadmat(os.path.join(ROOT_DIR, "Store/latent_dim.mat"))
    return inherent_dim[best_feature], best_feature
    
def get_labelled_features():
    """
        Method to create data of images by labels and different feature descriptor. With Cahing*
        ```
        {
            "label": { 
                "feature name": np.ndarray()
                ...
            ...
            },
        }
        ```
        Args:
            None
        Returned
            For each label and for each feature descriptor the images are stored in the form of np array 
            of size (n_samples, fd_size). fd_size is in the original space. 
    """

    data = loadmat(ROOT_DIR+'/Store/arrays.mat')
    labels = np.argmax(data["labels"], axis=1)
    features = ['cm_features', 'hog_features', 'avgpool_features', 'layer3_features', 'fc_features', 'resnet_features'] 

    label_features = {str(_label):{feature:None for feature in features} for _label in range(101)}

    print("Gathering label wise data...")
    for idx in tqdm(range(len(labels))):
        for feature in features:
            if not isinstance(label_features[str(labels[idx])][feature], np.ndarray):
                label_features[str(labels[idx])][feature] = data[feature][idx]
            else:
                label_features[str(labels[idx])][feature] = np.vstack((label_features[str(labels[idx])][feature], data[feature][idx]))
    savemat(os.path.join(ROOT_DIR, "Store/original_space_features_by_label.mat"), label_features)
    return label_features
    
def calculate_stress_per_label(label_features):
    """
        Method to calculate and create a DataFream consisting of labels, features and stress 
        for feature representation in original and latent space. The method creates and stores
        the DataFrame. It first normalizes the image data in original space and does dimentionality 
        reduction followed by stress calculatation. 
        Args:
            label_features(dict): map of labels and image features for each image in the following format:
            ```
            {
                "label": { 
                    "feature name": np.ndarray()
                    ...
                ...
                },
            }
            ```
        Return 
            pd.DataFrame consisting of label, stress in original space, stress in lantent space and stress differenece for \ 
            each feature descriptor. 
    
    """
    features = ['cm_features', 'hog_features', 'avgpool_features', 'layer3_features', 'fc_features', 'resnet_features']
    df = pd.DataFrame(columns=["Label", "Feature Space", "stress"])

    print("Dimentionality Reduction By Label...")
    for label in tqdm(range(101)):
        label = str(label)
        for feature in features:
            ## Calculate Stress for label & feature
            X_norm = featurenormalize(label_features[label][feature])

            X_latnet= pca_mle(X_norm)

            ## Calculate stress for latent features
            stress = calculate_stress(X_norm, X_latnet)
            ## Store
            stress_data = {"Label": label,"Feature Space": feature, "stress": stress}
            df = pd.concat([df, pd.DataFrame([stress_data])], ignore_index=True)

    df.to_csv(os.path.join(ROOT_DIR, "Store/stress_label.csv"), index = False)
    return df
    
def get_best_inherent_dim_per_label(label_features):
    """
        Method to find the best feature representation corresponding to every label.
        Args:
            label_features(dict): map of labels and image features for each image in the following format:
            ```
            {
                "label": { 
                    "feature name": np.ndarray()
                    ...
                ...
                },
            }
            ```
        Returns:
            list[tuple] consisting of label and feature name which represents the 'best' fd. 
    """
    df = calculate_stress_per_label(label_features)
    best_feature_per_label = []
    for label in range(101):
        label = str(label)
        df_label = df[df.Label == label]
        feature = df_label[df_label["stress"] == df_label["stress"].min()]["Feature Space"].values[0]
        best_feature_per_label.append((label, feature))
    return best_feature_per_label


def get_inherent_dim_label():
    """
        Method to get the inherent representation corresponding to each label. 
        Args:
            None
        Return:
            dict consisting of keys as labels and values as a tuples of best feature name and latent representation. 
    """
    label_features = get_labelled_features()
    best_feature_per_label = get_best_inherent_dim_per_label(label_features)
    
    inherent_dim_labels = dict()
    for label, feature in best_feature_per_label:
        inherent_dim_labels[label] = (feature, label_features[label][feature])
    return inherent_dim_labels
    
