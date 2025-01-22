import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
from numpy.random import uniform
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
from sklearn.decomposition import LatentDirichletAllocation
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
from scipy import linalg
from math import sqrt
from FeatureDescriptors.SimilarityScoreUtils import *
from Utilities.DisplayUtils import *
import streamlit as st
from pathlib import Path

def load_pickle_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        return None


def color_moments_calculator(image):
    
    if image.shape[2] == 2:
        # If the image has 2 channels (e.g., grayscale), convert it to 3 channels (RGB)
        print("2 channels")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize image to 300x100
    image = cv2.resize(image, (300, 100))
    
    # Partition the image into 10x10 grid
    rows = np.vsplit(image, 10)
    cells = [np.hsplit(row, 10) for row in rows]
    cells = np.array(cells).reshape(-1, 100, 3)

    # Initialize feature descriptor array
    color_moments = np.zeros((10, 10, 3, 3))

    # Compute color moments for each cell
    for row in range(10):
        for col in range(10):
            for ch in range(3):
                channel = cells[row*10 + col][:, ch]
                mean = np.mean(channel)
                std_dev = np.std(channel)
                skewness = skew(channel)
                color_moments[row, col, ch] = [mean, std_dev, skewness]

    return color_moments

    
#HOG Calculations

def hog_calculator(image):
    # Convert the image to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to 300x100
    image_resized = cv2.resize(image_grayscale, (300, 100))

    # Define the cell size
    cell_size = (30, 10)

    # Initialize the HOG descriptor array
    hog = []

    # Compute HOG descriptor for each cell
    for row in range(10):
        for col in range(10):
            # Extract the cell
            cell = image_resized[row*cell_size[1]:(row+1)*cell_size[1], col*cell_size[0]:(col+1)*cell_size[0]]

            # Calculate gradients using Sobel
            gradient_x = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=5)
            gradient_y = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=5)

            # Calculate gradient magnitude and orientation
            magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

            # Calculate the histogram for this cell
            hist, _ = np.histogram(angle, bins=9, range=(0, 180), weights=magnitude)

            # Append the histogram values to the HOG descriptor
            hog.extend(hist)

    return np.array(hog)

#ResNet Calculator
resnet_model = resnet50(pretrained=True)
def avgpool_calculator(image):
    
    # Define a hook to get the output of the "avgpool" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "avgpool" layer
    resnet_model.avgpool.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 2048-dimensional vector
    def get_2048_dimensional_vector(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze().numpy()

    # Get the 2048-dimensional vector
    vector_2048 = get_2048_dimensional_vector(image)

    # Reduce dimensions to 1024 by averaging adjacent entries
    vector_1024 = [(v1 + v2) / 2 for v1, v2 in zip(vector_2048[::2], vector_2048[1::2])]

    return vector_1024  

def layer3_calculator(image):
    
    # Define a hook to get the output of the "layer3" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "layer3" layer
    resnet_model.layer3.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 1024x14x14 tensor
    def get_1024x14x14_tensor(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze()

    # Get the 1024x14x14 tensor
    tensor_1024x14x14 = get_1024x14x14_tensor(image)

    # Convert the tensor to a 1024-dimensional vector by averaging each 14x14 slice
    vector_1024 = torch.mean(tensor_1024x14x14.view(1024, -1), dim=1).numpy()

    return vector_1024

#ResNet FC 1000

def fc_calculator(image):
    
    # Define a hook to get the output of the "fc" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "fc" layer
    resnet_model.fc.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 1000-dimensional tensor
    def get_1000_dimensional_tensor(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze()

    # Get the 1000-dimensional tensor
    tensor_1000 = get_1000_dimensional_tensor(image)

    return tensor_1000

def fc_calculator_2(image):
    
    # Define a hook to get the output of the "fc" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "fc" layer
    resnet_model.fc.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 1000-dimensional tensor
    def get_1000_dimensional_tensor(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze()

    # Get the 1000-dimensional tensor
    tensor_1000 = get_1000_dimensional_tensor(image)

    activation = torch.nn.Softmax()
    output_tensor = activation(tensor_1000)

    return output_tensor

def resnet_features(image):
    # Remove the final classification layer
    print("in")
    resnet50 = torch.nn.Sequential(*list(resnet_model.children())[:-13])
    # Set the model to evaluation mode
    resnet50.eval()

    # Define the preprocessing transformations
    def preprocess_img(image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)

    tensor_1000 = preprocess_img(image)
    print(type(tensor_1000))
    print("Shape of output tensor: ", tensor_1000.shape)

    with torch.no_grad():
        features = resnet50(tensor_1000)
    
    print("Features shape: ", features.shape)

    return features.squeeze().numpy()

    
def descriptor_calculator(image, idx,caltech101):
    color_moments = color_moments_calculator(image)
    hog_descriptor = hog_calculator(image)
    avgpool_descriptor = avgpool_calculator(image)
    layer3_descriptor = layer3_calculator(image)
    fc_descriptor = fc_calculator(image)
    fc_softmax_descriptor = fc_calculator_2(image)
    return {
        '_id': idx,
        'label': caltech101.__getitem__(index=idx)[1],
        'image': image.tolist() if isinstance(image, np.ndarray) else image,  # Convert the image to a list for storage
        'color_moments': color_moments.tolist() if isinstance(color_moments, np.ndarray) else color_moments,
        'hog_descriptor': hog_descriptor.tolist() if isinstance(hog_descriptor, np.ndarray) else hog_descriptor,
        'avgpool_descriptor': avgpool_descriptor.tolist() if isinstance(avgpool_descriptor, np.ndarray) else avgpool_descriptor,
        'layer3_descriptor': layer3_descriptor.tolist() if isinstance(layer3_descriptor, np.ndarray) else layer3_descriptor,
        'fc_descriptor': fc_descriptor.tolist(),
        'fc_softmax_descriptor': fc_softmax_descriptor.tolist()
    }
    

def queryksimilar(index,k,odd_feature_collection,feature_collection,similarity_collection,dataset,feature_space = None):
    
    similarity_scores = similarity_collection.find_one({'_id': index})
    color_moments_similar = dict(sorted(similarity_scores["color_moments"].items(), key = lambda x: x[1])[:k])
    hog_similar = dict(sorted(similarity_scores["hog_descriptor"].items(), key = lambda x: x[1],reverse = True)[:k])
    avgpool_similar = dict(sorted(similarity_scores["avgpool_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])
    layer3_similar = dict(sorted(similarity_scores["layer3_descriptor"].items(), key = lambda x: x[1])[:k])
    fc_similar = dict(sorted(similarity_scores["fc_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])
    fc_softmax_similar = dict(sorted(similarity_scores["fc_softmax_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])
    
    if index%2==0:
        imagedata = feature_collection.find_one({'_id': index})
    else:
        imagedata = odd_feature_collection.find_one({'_id': index})
    
    image = np.array(imagedata['image'], dtype=np.uint8)

    if feature_space == None:
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_color_moments(np.array(imagedata['color_moments']))
        display_hog(imagedata['hog_descriptor'])
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Softmax - Cosine Similarity')
        show_ksimilar(fc_softmax_similar,feature_collection,"Similarity Score:")

    elif feature_space == "Color Moments":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_color_moments(np.array(imagedata['color_moments']))
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")

    elif feature_space == "Histograms of Oriented Gradients(HOG)":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_hog(imagedata['hog_descriptor'])
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-AvgPool-1024":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Layer3-1024":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")

    elif feature_space == "ResNet-FC-1000":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Softmax":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['fc_softmax_descriptor'],"Query Image FC Softmax Descriptor")
        st.markdown('ResNet-Softmax - Cosine Similarity')
        show_ksimilar(fc_softmax_similar,feature_collection,"Similarity Score:")


    return similarity_scores

def queryksimilar_newimg(image, k,odd_feature_collection,feature_collection,similarity_collection,dataset,feature_space = None):

    color_moments = color_moments_calculator(image)
    hog_descriptor = hog_calculator(image)
    avgpool_descriptor = avgpool_calculator(image)
    layer3_descriptor = layer3_calculator(image)
    fc_descriptor = fc_calculator(image)
    fc_softmax_descriptor = fc_calculator_2(image)

    imagedata = {
        'image': image.tolist() if isinstance(image, np.ndarray) else image,  # Convert the image to a list for storage
        'color_moments': color_moments.tolist() if isinstance(color_moments, np.ndarray) else color_moments,
        'hog_descriptor': hog_descriptor.tolist() if isinstance(hog_descriptor, np.ndarray) else hog_descriptor,
        'avgpool_descriptor': avgpool_descriptor.tolist() if isinstance(avgpool_descriptor, np.ndarray) else avgpool_descriptor,
        'layer3_descriptor': layer3_descriptor.tolist() if isinstance(layer3_descriptor, np.ndarray) else layer3_descriptor,
        'fc_descriptor': fc_descriptor.tolist(),
        'fc_softmax_descriptor': fc_softmax_descriptor.tolist(),
    }
    similarity_scores = similarity_calculator_newimg(imagedata,odd_feature_collection,feature_collection,similarity_collection,dataset)
    color_moments_similar = dict(sorted(similarity_scores["color_moments"].items(), key = lambda x: x[1])[:k])
    hog_similar = dict(sorted(similarity_scores["hog_descriptor"].items(), key = lambda x: x[1])[-k:])
    avgpool_similar = dict(sorted(similarity_scores["avgpool_descriptor"].items(), key = lambda x: x[1])[-k:])
    layer3_similar = dict(sorted(similarity_scores["layer3_descriptor"].items(), key = lambda x: x[1])[:k])
    fc_similar = dict(sorted(similarity_scores["fc_descriptor"].items(), key = lambda x: x[1])[-k:])
    fc_softmax_similar = dict(sorted(similarity_scores["fc_softmax_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])

    if feature_space == None:
        display_color_moments(np.array(imagedata['color_moments']))
        display_hog(imagedata['hog_descriptor'])
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Softmax - Cosine Similarity')
        show_ksimilar(fc_softmax_similar,feature_collection,"Similarity Score:")

    elif feature_space == "Color Moments":
        display_color_moments(np.array(imagedata['color_moments']))
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")

    elif feature_space == "Histograms of Oriented Gradients(HOG)":
        display_hog(imagedata['hog_descriptor'])
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-AvgPool-1024":
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Layer3-1024":
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")

    elif feature_space == "ResNet-FC-1000":
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Softmax":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str("User Uploaded"))
        display_feature_vector(imagedata['fc_softmax_descriptor'],"Query Image FC Softmax Descriptor")
        st.markdown('ResNet-Softmax - Cosine Similarity')
        show_ksimilar(fc_softmax_similar,feature_collection,"Similarity Score:")

    return similarity_scores

def manual_svd(A):

    A = np.array(A)
    # Compute A*A^T and A^T*A
    AAT = np.dot(A, A.T)
    ATA = np.dot(A.T, A)

    # Compute the eigenvalues and eigenvectors of A*A^T
    eigenvalues_U, U = np.linalg.eigh(AAT)

    # Sort eigenvectors and eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues_U)[::-1]
    eigenvalues_U = eigenvalues_U[sorted_indices]
    U = U[:, sorted_indices]

    # Compute the singular values and their inverse from the eigenvalues
    singular_values = np.sqrt(eigenvalues_U)
    sigma_inv = 1 / singular_values

    # Compute V
    V = np.dot(U.T, A)

    return U, V, sigma_inv

def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    X = np.array(X)
    eps = 1e-5
    #print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    mask = np.sign(X)
    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = np.dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = np.dot(masked_X, Y.T)
        bottom = (np.dot((mask * (np.dot(A, Y))), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = np.dot(A.T, masked_X)
        bottom = np.dot(A.T, (mask * (np.dot(A, Y)))) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            #print 'Iteration {}:'.format(i),
            X_est = np.dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            #print 'fit residual', np.round(fit_residual, 4),
            #print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    YT = np.array(Y[:,:A.shape[1]]).T
    #print("Return Shape: "+str(A.shape)+" "+str(YT.shape))
    return np.dot(A,YT)

def kmeans_decomposition(X, k, max_iterations=100):
    X = np.array(X)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Assign each point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)

    # Compute distances to centroids as the decomposition
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    return distances

def reduce_dimensionality(feature_model, k, technique):
    if technique == 'SVD':
        U, V, sigma_inv = manual_svd(feature_model)

        print(U.shape,V.shape,sigma_inv.shape)

        # Take the first k columns of U and V
        latent_semantics = np.dot(U[:,:V.shape[1]], np.dot(np.diag(sigma_inv[:k]),V[:k, :]).T)

        #latent_semantics = latent_semantics[:,:k]

        print("Latent Semantics Shape: "+str(latent_semantics.shape))

        return latent_semantics

    elif technique == 'NNMF':

        latent_semantics_nnmf = nmf(feature_model, k, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6)

        print("Latent Semantics Shape: "+str(latent_semantics_nnmf.shape))

        return latent_semantics_nnmf

    elif technique == 'LDA':

        reducer = LatentDirichletAllocation(n_components=k)

        print("Transforming LDA")

        latent_semantics = reducer.fit_transform(feature_model)

        print("Latent Semantics Shape: "+str(latent_semantics.shape))

        return latent_semantics

    elif technique == 'k-Means':

        latent_semantics_kmeans = kmeans_decomposition(feature_model, k)

        print("Latent Semantics Shape: "+str(latent_semantics_kmeans.shape))

        return latent_semantics_kmeans

    else:
        raise ValueError("Invalid dimensionality reduction technique")

def get_top_k_latent_semantics(latent_semantics, k):
    top_k_indices = np.argsort(latent_semantics.sum(axis=1))[::-1][:k]
    return top_k_indices
    
def list_imageID_weight_pairs(top_k_indices, latent_semantics):
    imageID_weight_pairs = list(zip(top_k_indices, latent_semantics[:, top_k_indices]))
    imageID_weight_pairs.sort(key=lambda x: np.mean(x[1]), reverse=True)
    return imageID_weight_pairs

def ls1(feature_model,k,dimred,feature_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    try:

        data = scipy.io.loadmat(output_file+'arrays.mat')
        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(output_file,feature_collection)

        data = scipy.io.loadmat(output_file+'arrays.mat')

        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    feature_descriptors_array = []

    if feature_model == "Color Moments":

        output_file += "latent_semantics_1_color_moments_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(cm_features)

        print(feature_descriptors_array.shape)

    elif feature_model == "Histograms of Oriented Gradients(HOG)":

        output_file += "latent_semantics_1_hog_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(hog_features)

        print(feature_descriptors_array.shape)

    elif feature_model == "ResNet-AvgPool-1024":

        output_file += "latent_semantics_1_avgpool_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(avgpool_features)

        print(feature_descriptors_array.shape)

    elif feature_model == "ResNet-Layer3-1024":

        output_file += "latent_semantics_1_layer3_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(layer3_features)

        print(feature_descriptors_array.shape)

    elif feature_model == "ResNet-FC-1000":

        output_file += "latent_semantics_1_fc_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(fc_features)

        print(feature_descriptors_array.shape)

    elif feature_model == "RESNET":

        output_file += "latent_semantics_1_resnet_"+str(dimred)+"_"+str(k)+"_output.pkl"

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(resnet_features)

        print(feature_descriptors_array.shape)

    min_max_scaler = p.MinMaxScaler() 
    feature_descriptors_array = min_max_scaler.fit_transform(feature_descriptors_array)
    latent_semantics = reduce_dimensionality(feature_descriptors_array, k, dimred)
    top_k_indices = get_top_k_latent_semantics(latent_semantics, k)

    pickle.dump((top_k_indices,latent_semantics), open(output_file, 'wb+'))

    imageID_weight_pairs = list_imageID_weight_pairs(top_k_indices, latent_semantics)

    with st.container():
        rank = 1
        for imageID, weight in imageID_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Image ID: "+str(imageID)+" weights:"):
                st.write(weight.tolist())
            rank+=1


################## Task 5 Methods
def get_index_for_label(label, dataset):
    index = []
    for i in range(0, len(dataset.y), 2):
        if dataset.y[i] == label:
            index.append(i)
    
    return index

def get_sim_for_labels(labelx, labely, feature_model, odd_feature_collection, feature_collection, similarity_collection, dataset):
    scores = []

    feature_model_map = {"Color Moments": "color_moments", "Histograms of Oriented Gradients(HOG)": "hog_descriptor", 
                         "ResNet-AvgPool-1024": "avgpool_descriptor","ResNet-Layer3-1024": "layer3_descriptor","ResNet-FC-1000": "fc_descriptor", "RESNET":"fc_softmax_descriptor"}
    for x in labelx:
        sim_scores_for_x = similarity_collection.find_one({'_id': x})[feature_model_map.get(feature_model)]

        for y in labely:
            scores.append(sim_scores_for_x[str(y)])
    
    return np.mean(scores)

def get_labels_similarity_matrix(feature_model, odd_feature_collection, feature_collection, similarity_collection, dataset):
    labels = [label for label in range(101)]
    
    label_sim_matrix = np.nan * np.zeros((101,101))
    print(label_sim_matrix.shape)

    for idx in range(101):
        label_sim_matrix[idx][idx] = 1

    for labelx in tqdm(labels):
        labelx_idx = get_index_for_label(labelx, dataset)

        for labely in labels:
            if labelx == labely: continue
            if np.isnan(label_sim_matrix[labelx][labely]):

                labely_idx = get_index_for_label(labely, dataset)
                score = get_sim_for_labels(labelx_idx, labely_idx, feature_model, odd_feature_collection, feature_collection, similarity_collection, dataset)
                label_sim_matrix[labelx][labely] = label_sim_matrix[labely][labelx] = score

        # print("Label Similarities for Label "+str(labelx)+" is of len: "+str(len(label_sim_matrix[labelx]))+" and has values: "+str(label_sim_matrix[labelx]))

    print(label_sim_matrix)
    
    return label_sim_matrix

def get_reduced_dim_labels(sim_matrix, dimred, k):
    latent_semantics = reduce_dimensionality(sim_matrix, k, dimred)
    top_k_label_indices = get_top_k_latent_semantics(latent_semantics, k)

    return latent_semantics, top_k_label_indices

def list_label_weight_pairs(top_k_indices, latent_semantics):
    label_weight_pairs = list(zip(top_k_indices, latent_semantics[:, top_k_indices]))
    label_weight_pairs.sort(key=lambda x: np.mean(x[1]), reverse=True)
    
    with st.container():
        rank = 1
        for labelID, weight in label_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Label: "+str(labelID)+" weights:"):
                st.write(weight.tolist())
            rank+=1


def ls3(feature_model, dimred, k, odd_feature_collection, feature_collection, similarity_collection, caltech101, imageType='odd',saveLatentSemantic = 'False'):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    ### Creating Label-Label Sim Matx -1
    baseIndex = 1 if imageType == 'odd' else 0
    sim_matrix = get_labels_similarity_matrix(feature_model, odd_feature_collection, feature_collection, similarity_collection, caltech101,baseIndex) ##Labels should be in increasing order
    ### Dim reduction on Sim matx -2
    latent_semantics, top_k_indices = get_reduced_dim_labels(sim_matrix, dimred, k) 

    if saveLatentSemantic == 'True':
        ### Storing latent Semantics - 3
        if feature_model == "Color Moments":

            output_file += "latent_semantics_3_color_moments_"+str(dimred)+"_"+str(k)+"_output.pkl"

        elif feature_model == "Histograms of Oriented Gradients(HOG)":

            output_file += "latent_semantics_3_hog_"+str(dimred)+"_"+str(k)+"_output.pkl"

        elif feature_model == "ResNet-AvgPool-1024":

            output_file += "latent_semantics_3_avgpool_"+str(dimred)+"_"+str(k)+"_output.pkl"

        elif feature_model == "ResNet-Layer3-1024":

            output_file += "latent_semantics_3_layer3_"+str(dimred)+"_"+str(k)+"_output.pkl"
        elif feature_model == "ResNet-FC-1000":

            output_file += "latent_semantics_3_fc_"+str(dimred)+"_"+str(k)+"_output.pkl"

        elif feature_model == "RESNET":

            output_file += "latent_semantics_3_resnet_"+str(dimred)+"_"+str(k)+"_output.pkl"

        pickle.dump((top_k_indices,latent_semantics), open(output_file, 'wb+'))
        ### Listing Label Weight Pairs - 4
        list_label_weight_pairs(top_k_indices, latent_semantics)

    return latent_semantics , top_k_indices


#################################
def get_class_name(label):
    data = {
	0: "Faces", 1: "Faces_easy", 2: "Leopards", 3: "Motorbikes", 4: "accordion", 5: "airplanes", 6: "anchor", 7: "ant", 8: "barrel", 9: "bass",
	10: "beaver", 11: "binocular", 12: "bonsai", 13: "brain", 14: "brontosaurus", 15: "buddha", 16: "butterfly", 17: "camera", 18: "cannon", 19: "car_side",
	20: "ceiling_fan", 21: "cellphone", 22: "chair", 23: "chandelier", 24: "cougar_body", 25: "cougar_face", 26: "crab", 27: "crayfish", 28: "crocodile", 29: "crocodile_head",
	30: "cup", 31: "dalmatian", 32: "dollar_bill", 33: "dolphin", 34: "dragonfly", 35: "electric_guitar", 36: "elephant", 37: "emu", 38: "euphonium", 39: "ewer",
	40: "ferry", 41: "flamingo", 42: "flamingo_head", 43: "garfield", 44: "gerenuk", 45: "gramophone", 46: "grand_piano", 47: "hawksbill", 48: "headphone", 49: "hedgehog",
	50: "helicopter", 51: "ibis", 52: "inline_skate", 53: "joshua_tree", 54: "kangaroo", 55: "ketch", 56: "lamp", 57: "laptop", 58: "llama", 59: "lobster",
	60: "lotus", 61: "mandolin", 62: "mayfly", 63: "menorah", 64: "metronome", 65: "minaret", 66: "nautilus", 67: "octopus", 68: "okapi", 69: "pagoda",
	70: "panda", 71: "pigeon", 72: "pizza", 73: "platypus", 74: "pyramid", 75: "revolver", 76: "rhino", 77: "rooster", 78: "saxophone", 79: "schooner",
	80: "scissors", 81: "scorpion", 82: "seahorse", 83: "snoopy", 84: "soccer_ball", 85: "stapler", 86: "starfish", 87: "stegosaurus", 88: "stop_sign", 89: "strawberry",
	90: "sunflower", 91: "tick", 92: "trilobite", 93: "umbrella", 94: "watch", 95: "water_lilly", 96: "wheelchair", 97: "wild_cat", 98: "windsor_chair", 99: "wrench", 100: "yin_yang"}

    return data[label]

def CPDecomposition(cp_tensor,k,max_iter=1):
    print("Calculating CP_Decomposition")

    A = np.random.random((k, cp_tensor.shape[0]))
    B = np.random.random((k, cp_tensor.shape[1]))
    C = np.random.random((k, cp_tensor.shape[2]))

    for epoch in tqdm(range(max_iter)):
        # optimize a
        A_Input = khatri_rao([B.T, C.T])
        A_Target = tl.unfold(cp_tensor, mode=0).T
        A = np.linalg.solve(A_Input.T.dot(A_Input), A_Input.T.dot(A_Target))

        # optimize b
        B_Input = khatri_rao([A.T, C.T])
        B_Target = tl.unfold(cp_tensor, mode=1).T
        B = np.linalg.solve(B_Input.T.dot(B_Input), B_Input.T.dot(B_Target))

        # optimize c
        C_Input = khatri_rao([A.T, B.T])
        C_Target = tl.unfold(cp_tensor, mode=2).T
        C = np.linalg.solve(C_Input.T.dot(C_Input), C_Input.T.dot(C_Target))

    A,B,C=A.T,B.T,C.T
    print(A.shape,B.shape,C.shape)
    return C

def get_top_k_cp_indices(label_factors, k):
    top_k_indices = np.argsort(-label_factors.sum(axis=1))[:k]
    return top_k_indices

def list_label_weight_pairs_cp(top_k_indices,label_factors):
    label_weight_pairs = []

    for i in top_k_indices:
        label_weight_pairs.append((i, label_factors[i])) 

    return label_weight_pairs

def store_by_feature(output_file,feature_collection):

    labels = []
    cm_features = []
    hog_features = []
    avgpool_features =[]
    layer3_features = []
    fc_features = []
    resnet_features = []

    for index in tqdm(range(0,dataset_size,2)):
        doc = feature_collection.find_one({'_id': index})

        label = int(doc['label'])
        print(label)

        labelarray = [0 if x!=label else 1 for x in range(101)]

        labels.append(labelarray)
                    
        fetchedarray = doc['color_moments']

        cmarray = []

        for row in range(0,10):
            for col in range(0,10):
                for channel in fetchedarray[row][col]:
                    cmarray.append(channel[0])
                    cmarray.append(channel[1])
                    cmarray.append(channel[2])

        cmarray = [0 if pd.isna(x) else x for x in cmarray]

        cmarray = np.array(cmarray)

        cm_features.append(cmarray)

        fetchedarray = doc['hog_descriptor']
                
        hogarray = [0 if pd.isna(x) else x for x in fetchedarray]

        hogarray = np.array(hogarray)

        hog_features.append(hogarray)

        fetchedarray = doc['avgpool_descriptor']
                
        avgpoolarray = [0 if pd.isna(x) else x for x in fetchedarray]

        avgpoolarray = np.array(avgpoolarray)

        avgpool_features.append(avgpoolarray)

        fetchedarray = doc['layer3_descriptor']

        layer3array = [0 if pd.isna(x) else x for x in fetchedarray]

        layer3array = np.array(layer3array)

        layer3_features.append(layer3array)

        fetchedarray = doc['fc_descriptor']

        fcarray = [0 if pd.isna(x) else x for x in fetchedarray]

        fcarray = np.array(fcarray)

        fc_features.append(fcarray)

        fetchedarray = doc['fc_softmax_descriptor']

        resnetarray = [0 if pd.isna(x) else x for x in fetchedarray]

        resnetarray = np.array(resnetarray)

        resnet_features.append(resnetarray)

    scipy.io.savemat(output_file+'arrays.mat', {'labels': labels, 'cm_features': cm_features, 'hog_features':hog_features, 'avgpool_features': avgpool_features,'layer3_features':layer3_features, 'fc_features': fc_features, 'resnet_features':resnet_features})  

def store_by_feature_odd(output_file,feature_collection):

    labels = []
    cm_features = []
    hog_features = []
    avgpool_features =[]
    layer3_features = []
    fc_features = []
    resnet_features = []

    for index in tqdm(range(1,dataset_size,2)):
        doc = feature_collection.find_one({'_id': index})

        label = int(doc['label'])
        print(label)

        labelarray = [0 if x!=label else 1 for x in range(101)]

        labels.append(labelarray)
                    
        fetchedarray = doc['color_moments']

        cmarray = []

        for row in range(0,10):
            for col in range(0,10):
                for channel in fetchedarray[row][col]:
                    cmarray.append(channel[0])
                    cmarray.append(channel[1])
                    cmarray.append(channel[2])

        cmarray = [0 if pd.isna(x) else x for x in cmarray]

        cmarray = np.array(cmarray)

        cm_features.append(cmarray)

        fetchedarray = doc['hog_descriptor']
                
        hogarray = [0 if pd.isna(x) else x for x in fetchedarray]

        hogarray = np.array(hogarray)

        hog_features.append(hogarray)

        fetchedarray = doc['avgpool_descriptor']
                
        avgpoolarray = [0 if pd.isna(x) else x for x in fetchedarray]

        avgpoolarray = np.array(avgpoolarray)

        avgpool_features.append(avgpoolarray)

        fetchedarray = doc['layer3_descriptor']

        layer3array = [0 if pd.isna(x) else x for x in fetchedarray]

        layer3array = np.array(layer3array)

        layer3_features.append(layer3array)

        fetchedarray = doc['fc_descriptor']

        fcarray = [0 if pd.isna(x) else x for x in fetchedarray]

        fcarray = np.array(fcarray)

        fc_features.append(fcarray)

    scipy.io.savemat(output_file+'arrays_odd.mat', {'labels': labels, 'cm_features': cm_features, 'hog_features':hog_features, 'avgpool_features': avgpool_features,'layer3_features':layer3_features, 'fc_features': fc_features})  

def ls2(feature_model,k,feature_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    try:

        data = scipy.io.loadmat(output_file+'arrays.mat')
        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    except scipy.io.matlab.miobase.MatReadError as e:

        store_by_feature(output_file,feature_collection)

        data = scipy.io.loadmat(output_file+'arrays.mat')

        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    

    print(type(labels),type(cm_features),type(hog_features),type(avgpool_features),type(layer3_features),type(fc_features),type(resnet_features))
    print(np.array(labels).shape,np.array(cm_features).shape,np.array(hog_features).shape,np.array(avgpool_features).shape,np.array(layer3_features).shape,np.array(fc_features).shape,np.array(resnet_features).shape)

    label = tl.tensor(labels)
    cm_features = tl.tensor(cm_features)
    hog_features = tl.tensor(hog_features)
    layer3_features = tl.tensor(layer3_features)
    fc_features = tl.tensor(fc_features)
    resnet_features - tl.tensor(resnet_features)

    if feature_model == "Color Moments":

        output_file += "latent_semantics_2_color_moments_"+str(k)+"_output.pkl"

        num_samples = len(cm_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(cm_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(cm_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)

    elif feature_model == "Histograms of Oriented Gradients(HOG)":

        output_file += "latent_semantics_2_hog_"+str(k)+"_output.pkl"

        num_samples = len(hog_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(cm_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(hog_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)

    elif feature_model == "ResNet-AvgPool-1024":

        output_file += "latent_semantics_2_avgpool_"+str(k)+"_output.pkl"

        num_samples = len(avgpool_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(avgpool_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(avgpool_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)

    elif feature_model == "ResNet-Layer3-1024":

        output_file += "latent_semantics_2_layer3_"+str(k)+"_output.pkl"

        num_samples = len(layer3_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(layer3_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(layer3_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)

    elif feature_model == "ResNet-FC-1000":

        output_file += "latent_semantics_2_fc_"+str(k)+"_output.pkl"

        num_samples = len(fc_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(fc_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(fc_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)

    elif feature_model == "RESNET":

        output_file += "latent_semantics_2_resnet_"+str(k)+"_output.pkl"

        num_samples = len(resnet_features)  # Adjust based on your data
        num_labels = 101
        descriptor_length = len(resnet_features[0])  # Adjust based on your feature descriptor length

        # Initialize arrays to hold feature descriptors and labels
        feature_descriptors_array = np.array(resnet_features)
        labels_array = np.array(labels).reshape(-1, 101, 1)

        print(feature_descriptors_array.shape,labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        result_list = []

        # Iterate through each image
        for i in tqdm(range(4339)):
            # Extract the feature descriptors and labels for the current image
            feature_descriptors_image = feature_descriptors_array[i]
            labels_image = labels_array[i]

            feature_descriptors_image = (feature_descriptors_image - np.mean(feature_descriptors_image)) / np.std(feature_descriptors_image)

            feature_descriptors_image = np.nan_to_num(feature_descriptors_image)

            # Perform the np.multiply.outer operation
            cp_tensor = np.multiply.outer(feature_descriptors_image, labels_image)

            # Append the result to the list
            result_list.append(cp_tensor)

        # Convert the list of tensors to a NumPy array
        result_array = np.array(result_list)
        cp_tensor = np.squeeze(result_array, axis=3)

        # The resulting array will have shape (4339, 900, 101)
        print(cp_tensor.shape)

        label_factors = CPDecomposition(cp_tensor,k,max_iter=10)

        print(label_factors.shape)


    top_k_indices = get_top_k_cp_indices(label_factors, k)

    print(top_k_indices,top_k_indices.shape)

    pickle.dump((top_k_indices, label_factors), open(output_file, 'wb+'))

    label_weight_pairs = list_label_weight_pairs_cp(top_k_indices,label_factors)

    with st.container():
        rank = 1
        for label, weight in label_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Label No: "+str(label)+" Label: "+get_class_name(label)+" weights:"):
                st.write(weight.tolist())
            rank+=1

def ls4(feature_model,k,dimred,similarity_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    similarity_matrix = [[0 for col in range(4339)] for row in range(4339)]

    if feature_model == "Color Moments":
        output_file += "latent_semantics_4_color_moments_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if 1 - scores['color_moments'][str(cmpidx)]<0:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 0
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1 - scores['color_moments'][str(cmpidx)]


    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        output_file += "latent_semantics_4_hog_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if scores['hog_descriptor'][str(cmpidx)]>1:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['hog_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-AvgPool-1024":
        output_file += "latent_semantics_4_avgpool_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if scores['avgpool_descriptor'][str(cmpidx)]>1:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['avgpool_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-Layer3-1024":
        output_file += "latent_semantics_4_layer3_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if 1 - scores['layer3_descriptor'][str(cmpidx)]<0:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 0
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1 - scores['layer3_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-FC-1000":
        output_file += "latent_semantics_4_fc_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if scores['fc_descriptor'][str(cmpidx)]>1:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['fc_descriptor'][str(cmpidx)]

    elif feature_model == "RESNET":
        output_file += "latent_semantics_4_resnet_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                if scores['fc_softmax_descriptor'][str(cmpidx)]>1:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1
                else:
                    similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['fc_softmax_descriptor'][str(cmpidx)]

    similarity_vector = np.array(similarity_matrix).reshape(-1,1)
    #print(similarity_vector.shape)
    latent_semantics = reduce_dimensionality(similarity_matrix, k, dimred)
    top_k_indices = get_top_k_latent_semantics(latent_semantics, k)

    pickle.dump((top_k_indices, latent_semantics), open(output_file, 'wb+'))

    imageID_weight_pairs = list_imageID_weight_pairs(top_k_indices, latent_semantics)

    with st.container():
        rank = 1
        for imageID, weight in imageID_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Image ID: "+str(imageID)+" weights:"):
                st.write(weight.tolist())
            rank+=1

    return similarity_matrix

def get_similar_ls(idx,latsem, feature_model, latentk, dimred,k,uploaded_file,feature_collection):
    mod_path = Path(__file__).parent.parent
    pkl_file_path = str(mod_path)+"/LatentSemantics/"
    
    if feature_model == "Color Moments":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+str(latentk)+"_output.pkl"
    

    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_"+str(latentk)+"_output.pkl"    
        

    elif feature_model == "ResNet-AvgPool-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_avgpool_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_avgpool_"+str(latentk)+"_output.pkl"
        

    elif feature_model == "ResNet-Layer3-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_"+str(latentk)+"_output.pkl"
       
    elif feature_model == "ResNet-FC-1000":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_"+str(latentk)+"_output.pkl"            

    elif feature_model == "RESNET":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+str(latentk)+"_output.pkl"            

    
    print(pkl_file_path)
    
    #pkl_file_path+="latent_semantics_4_layer3_descriptor_LDA_5_output.pkl"
    with open(pkl_file_path,'rb') as file:
        print('File path is '+pkl_file_path)
        __,pickle_data = pickle.load(file)
        
    mat_file_path = str(mod_path)+"/LatentSemantics/"

    try:

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')
        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(mat_file_path,feature_collection)

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')

        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']
    
    
    
    print('Pickle File Loaded')
        
    print(pickle_data.shape)
    
    print(labels.shape , labels)

    if uploaded_file == None:

        query_label_index = np.nonzero(labels[int(idx/2)])[0][0]
         
        query_img_label = get_class_name(np.nonzero(labels[int(idx/2)])[0][0])
            
        print(query_img_label)
            
        if(latsem == 'LS1' or latsem == 'LS4'):
            get_ls_similar_labels_image_weighted(pickle_data,labels,idx, k)
                
        else:
            get_ls_similar_labels_label_weighted(pickle_data, query_label_index, k)         

    else:

        #Calculate Feature Descriptor for input image and reduce dimensions
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

              
        image = cv2.cvtColor(opencv_image,cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA) 

        image = np.array(image)

        if feature_model == "Color Moments":

            input_image_feature_descriptor = color_moments_calculator(image)

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape)

            input_image_feature_descriptors = np.insert(cm_features, 0, input_image_feature_descriptor, axis=0)

        elif feature_model == "Histograms of Oriented Gradients(HOG)": 
            
            input_image_feature_descriptor = hog_calculator(image)

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape)

            input_image_feature_descriptors = np.insert(hog_features, 0, input_image_feature_descriptor, axis=0)

        elif feature_model == "ResNet-AvgPool-1024":

            input_image_feature_descriptor = avgpool_calculator(image)   

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape)

            input_image_feature_descriptors = np.insert(avgpool_features, 0, input_image_feature_descriptor, axis=0)

        elif feature_model == "ResNet-Layer3-1024":

            input_image_feature_descriptor = layer3_calculator(image) 

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape)

            input_image_feature_descriptors = np.insert(layer3_features, 0, input_image_feature_descriptor, axis=0)
           
        elif feature_model == "ResNet-FC-1000":

            input_image_feature_descriptor = fc_calculator(image) 

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape) 

            input_image_feature_descriptors = np.insert(fc_features, 0, input_image_feature_descriptor, axis=0)

        elif feature_model == "RESNET":

            input_image_feature_descriptor = fc_calculator_2(image)  

            input_image_feature_descriptor = np.array(input_image_feature_descriptor).reshape(1,-1)

            print(input_image_feature_descriptor.shape)

            input_image_feature_descriptors = np.insert(resnet_features, 0, input_image_feature_descriptor, axis=0)

        if dimred == 'SVD':

            latent_semantics_input_image = reduce_dimensionality(input_image_feature_descriptors, latentk, dimred)            

        elif dimred == 'NNMF':

            latent_semantics_input_image = reduce_dimensionality(input_image_feature_descriptors, latentk, dimred)

        elif dimred == 'LDA':

            latent_semantics_input_image = reduce_dimensionality(input_image_feature_descriptors, latentk, dimred)

        elif dimred == 'k-Means':

            latent_semantics_input_image = reduce_dimensionality(input_image_feature_descriptor, latentk, dimred)

        latent_semantics_input_image = latent_semantics_input_image[0].reshape(1,-1)

        print("Input Image Semantics: "+str(latent_semantics_input_image.shape))

        if(latsem == 'LS1' or latsem == 'LS4'):

            get_ls_similar_labels_image_weighted(pickle_data,labels,idx, k, latent_semantics_input_image)
                
        else:
            get_ls_similar_labels_label_weighted(pickle_data, latent_semantics_input_image, k,False,True)

def get_ls_similar_labels_label_weighted(pickle_data, query_label_index, k, getdict = False, input_image = False):
    
    if input_image == False:

        sim_la = {}
        for i in range(0,101):
            sim_la[i] = cosine_similarity_calculator(pickle_data[i],pickle_data[query_label_index])
            #print(sim_la[i])
        
        sim_la = dict(sorted(sim_la.items(), key = lambda x: x[1] , reverse = True)[:k])

        if getdict == True:
            return sim_la
        
        #print top k matching labels
        for key, val in sim_la.items():
            st.write(get_class_name(key), ": ", val)

    else:

        sim_la = {}
        for i in range(0,101):
            sim_la[i] = cosine_similarity_calculator(pickle_data[i],query_label_index)
            #print(sim_la[i])
        
        sim_la = dict(sorted(sim_la.items(), key = lambda x: x[1] , reverse = True)[:k])

        if getdict == True:
            return sim_la
        
        #print top k matching labels
        for key, val in sim_la.items():
            st.write(get_class_name(key), ": ", val)
        
        
def get_ls_similar_labels_image_weighted(pickle_data,labels, idx, k, latent_semantics_input_image = None):

    similarity_image_scores = {}

    if latent_semantics_input_image == None:
        
        for i in range(0,8677,2):
            if get_class_name(np.nonzero(labels[int(i/2)])[0][0]) not in similarity_image_scores.keys():
                similarity_image_scores[get_class_name(np.nonzero(labels[int(i/2)])[0][0])]=[]
            similarity_image_scores[get_class_name(np.nonzero(labels[int(i/2)])[0][0])].append(cosine_similarity_calculator(pickle_data[int(i/2)],pickle_data[int(idx/2)]))
        
        
        sim_la = {}
        for key in similarity_image_scores.keys():
            sim_la[key] = np.mean(similarity_image_scores[key])
        
        sim_la = dict(sorted(sim_la.items(), key = lambda x: x[1] , reverse = True)[:k])
        
        #print top k matching labels
        for key, val in sim_la.items():
            st.write(key, ": ", val)

    else:

        for i in range(0,8677,2):
            if get_class_name(np.nonzero(labels[int(i/2)])[0][0]) not in similarity_image_scores.keys():
                similarity_image_scores[get_class_name(np.nonzero(labels[int(i/2)])[0][0])]=[]
            similarity_image_scores[get_class_name(np.nonzero(labels[int(i/2)])[0][0])].append(cosine_similarity_calculator(pickle_data[int(i/2)],latent_semantics_input_image))
        
        
        sim_la = {}
        for key in similarity_image_scores.keys():
            sim_la[key] = np.mean(similarity_image_scores[key])
        
        sim_la = dict(sorted(sim_la.items(), key = lambda x: x[1] , reverse = True)[:k])
        
        #print top k matching labels
        for key, val in sim_la.items():
            st.write(key, ": ", val)

def get_ls_similar_images_from_label_image_weighted(pickle_data,label, k,feature_collection):

    similarity_image_scores = {}
    similarity_label_scores = {}

    image_data_by_label = feature_collection.find({'label':label})

    final_scores = []

    required_indices_for_label = []
    labels_array = []
    
    #Segregate images of the label in particular, and calculate similarity scores
    for doc in tqdm(image_data_by_label):
        required_indices_for_label.append(doc['_id'])
        labels_array.append(doc['label'])

    print("Labels array"+str(type(labels_array))+str(len(labels_array)))

    label_ls = []

    for index in required_indices_for_label:
        label_ls.append(pickle_data[int(index/2)])

    label_ls = np.array(label_ls)
    print(label_ls.shape)

    mean_label_ls = np.mean(label_ls,axis = 0).reshape(-1,1)
    print(mean_label_ls.shape,np.array(pickle_data[0]).reshape(-1,1).shape)

    for index in tqdm(range(0,8677,2)):
        #print(int(index/2))
        imagedata = feature_collection.find_one({'_id':index})
        imglabel = imagedata['label']
        #print("Label"+str(label))
        similarity_image_scores[index] = cosine_similarity_calculator(pickle_data[int(index/2)].reshape(-1,1), mean_label_ls)
        if imglabel not in similarity_label_scores.keys():
            similarity_label_scores[imglabel] = []
            similarity_label_scores[imglabel].append(similarity_image_scores[index])
        else:
            similarity_label_scores[imglabel].append(similarity_image_scores[index])

    return similarity_image_scores,similarity_label_scores

def get_features_from_mat(data, feature_model):
    
    if "color" in feature_model or "cm" in feature_model: return data["cm_features"]
    elif "hog" in feature_model: return data["hog_features"]
    elif "avgpool" in feature_model: return data["avgpool_features"]
    elif "layer3" in feature_model: return data["layer3_features"]
    elif "fc" in feature_model: return data["fc_features"]
    elif "resnet" in feature_model: return data["resnet_features"]
        
    
def get_latent_semantics(pkl_file_path, latsem,latentk,dimred, feature_model):

    if feature_model == "Color Moments":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+str(latentk)+"_output.pkl"
    

    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_descriptor_"+str(latentk)+"_output.pkl"    
        

    elif feature_model == "ResNet-AvgPool-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_ResNet-AvgPool-1024_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_ResNet-AvgPool-1024_"+str(latentk)+"_output.pkl"
        

    elif feature_model == "ResNet-Layer3-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_descriptor_"+str(latentk)+"_output.pkl"
       
    elif feature_model == "ResNet-FC-1000":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_descriptor_"+str(latentk)+"_output.pkl"

    elif feature_model == "RESNET":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+str(latentk)+"_output.pkl" 

    with open(pkl_file_path,'rb') as file:
        print('File path is '+pkl_file_path)
        __,pickle_data = pickle.load(file)

    return pickle_data   


        
def get_topk_image_score(k, query_ls, latent_semantics, feature_model):
    scores = []
    sim_score_method = {"Color Moments": similarity_score_color_moments, "Histograms of Oriented Gradients(HOG)": similarity_score_hog,
                        "ResNet-AvgPool-1024": similarity_score_avgpool, "ResNet-Layer3-1024": similarity_score_layer3,
                        "ResNet-FC-1000": similarity_score_fc, "RESNET": get_similarity_score_resnet}
    
    for ls in latent_semantics:
        score = sim_score_method[feature_model](query_ls, ls)
        if "layer3" in feature_model or "color" in feature_model:
            score = 1-score
        scores.append(score)
    
    index =  np.argsort(scores)[::-1][:k]
    scores = [scores[idx] for idx in index]

    return index, scores

def get_simlar_ls(idx, feature_model, k,latsem, latentk, dimred, odd_feature_collection, feature_collection, similarity_collection,caltech101):
    
    mod_path = Path(__file__).parent.parent
    mat_file_path = str(mod_path)+"/LatentSemantics/"
    data = scipy.io.loadmat(mat_file_path+'arrays.mat')

    latent_semantics = get_latent_semantics(mat_file_path,latsem,latentk,dimred, feature_model)

    if latsem == "LS1" or latsem == "LS4":

        if idx%2==0:
            query_ls = latent_semantics[idx//2]
            _imagedata = feature_collection.find_one({'_id': idx})

        else:
            _imagedata = odd_feature_collection.find_one({"_id": idx})
            if feature_model == "RESNET":
                image = np.array(_imagedata['image'], dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA) 
                image = np.array(image)
                odd_feature = fc_calculator_2(image).reshape(1,-1)
            
            else:
                odd_feature = np.array(_imagedata[feature_model]).reshape(1,-1)
            ####getfeaturesforodd
            features = get_features_from_mat(data, feature_model)
            mixed_feature_descriptors = np.insert(features, 0, odd_feature, axis=0)
            query_ls = reduce_dimensionality(mixed_feature_descriptors, latentk, dimred)[0]


        top_k_index, scores = get_topk_image_score(k, query_ls, latent_semantics, feature_model)
        k_similar = {str(idx*2): score for idx, score in zip(top_k_index, scores)}
        ### Display Images and Score
        
        image = np.array(_imagedata['image'], dtype=np.uint8)
        display_image_centered(np.array(image),str(idx))
        show_ksimilar(k_similar, feature_collection, f"Most Similar {k} images with scores: ")

    else:

        if idx%2==0:
            _imagedata = feature_collection.find_one({'_id': idx})
            label = _imagedata['label']

        else:
            _imagedata = odd_feature_collection.find_one({"_id": idx})
            """if feature_model == "RESNET":
                                                    image = np.array(_imagedata['image'], dtype=np.uint8)
                                                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                                    image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA) 
                                                    image = np.array(image)
                                                    odd_feature = fc_calculator_2(image).reshape(1,-1)
                                                
                                                else:
                                                    odd_feature = np.array(_imagedata[feature_model]).reshape(1,-1)
                                                ####getfeaturesforodd
                                                features = get_features_from_mat(data, feature_model)
                                                mixed_feature_descriptors = np.insert(features, 0, odd_feature, axis=0)
                                                query_ls = reduce_dimensionality(mixed_feature_descriptors, latentk, dimred)[0]"""
            label = _imagedata['label']

        sim_la = get_ls_similar_labels_label_weighted(latent_semantics, label, 1, True)

        print(get_class_name(list(sim_la.keys())[0]))

        matching_label = list(sim_la.keys())[0]

        similarity_calculator_by_label(matching_label,feature_model,k,odd_feature_collection,feature_collection,similarity_collection,caltech101)



def get_simlar_ls_img(imagedata, feature_model, k, latsem, latentk, dimred, feature_collection) :

    mod_path = Path(__file__).parent.parent
    mat_file_path = str(mod_path)+"/LatentSemantics/"
    data = scipy.io.loadmat(mat_file_path+'arrays.mat')

    latent_semantics = get_latent_semantics(mat_file_path,latsem,latentk,dimred, feature_model)
    

    if feature_model == "Color Moments":
        odd_feature = np.array(imagedata['color_moments']).reshape(1,-1)
        
    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        odd_feature = np.array(imagedata['hog_descriptor']).reshape(1,-1)

    elif feature_model == "ResNet-AvgPool-1024":
        odd_feature = np.array(imagedata['avgpool_descriptor']).reshape(1,-1)
        

    elif feature_model == "ResNet-Layer3-1024":
        odd_feature = np.array(imagedata['layer3_descriptor']).reshape(1,-1)
       
    elif feature_model == "ResNet-FC-1000":
        odd_feature = np.array(imagedata['fc_descriptor']).reshape(1,-1)

    elif feature_model == "RESNET":
        odd_feature = fc_calculator_2(np.array(imagedata["image"], dtype=np.uint8)).reshape(1,-1)

    else:
        odd_feature = np.array(imagedata[feature_model]).reshape(1,-1)

    features = get_features_from_mat(data, feature_model)
    mixed_feature_descriptors = np.insert(features, 0, odd_feature, axis=0)
    query_ls = reduce_dimensionality(mixed_feature_descriptors, latentk, dimred)[0]

    top_k_index, scores = get_topk_image_score(k, query_ls, latent_semantics, feature_model)
    k_similar = {str(idx*2): score for idx, score in zip(top_k_index, scores)}
    
    ### Display Images and Score
    show_ksimilar(k_similar, feature_collection, f"Most Similar {k} images with scores: ")
    
    
    
def get_simlar_ls_label():
    print(" identifies and lists k most likely matching labels, along with their scores, under the selected latent space.")
def get_simlar_ls_label_img():
    print(" identifies and lists k most likely matching labels, along with their scores, under the selected latent space. for new image upload")

def get_simlar_ls__by_label(lbl, latsem, feature_model, latentk, dimred, k, feature_collection):
    #print("identifies and lists k most likely matching labels, along with their scores, under the selected latent space.")

    mod_path = Path(__file__).parent.parent
    pkl_file_path = str(mod_path)+"/LatentSemantics/"
    
    if feature_model == "Color Moments":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+str(latentk)+"_output.pkl"
    

    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_descriptor_"+str(latentk)+"_output.pkl"    
        

    elif feature_model == "ResNet-AvgPool-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_ResNet-AvgPool-1024_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_ResNet-AvgPool-1024_"+str(latentk)+"_output.pkl"
        

    elif feature_model == "ResNet-Layer3-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_descriptor_"+str(latentk)+"_output.pkl"
       
    elif feature_model == "ResNet-FC-1000":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_descriptor_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_descriptor_"+str(latentk)+"_output.pkl"

    elif feature_model == "RESNET":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+str(latentk)+"_output.pkl"

    with open(pkl_file_path,'rb') as file:
        print('File path is '+pkl_file_path)
        __,pickle_data = pickle.load(file)
        
    mat_file_path = str(mod_path)+"/LatentSemantics/"

    try:

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')
        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(mat_file_path,feature_collection)

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')

        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    print('Pickle File Loaded')
        
    print(pickle_data.shape)

    if(latsem == 'LS1' or latsem == 'LS4'):
        __, sim_label_image_dict = get_ls_similar_images_from_label_image_weighted(pickle_data,lbl, k,feature_collection)

        for i in sim_label_image_dict.keys():
            sim_label_image_dict[i] = sum(sim_label_image_dict[i])/len(sim_label_image_dict[i])

        sim_label_image_dict = dict(sorted(sim_label_image_dict.items(), key = lambda x: x[1], reverse=True)[:k])

        #print top k matching labels
        for key, val in sim_label_image_dict.items():
            st.write(get_class_name(key), ": ", val)
            st.write("")   

        return sim_label_image_dict
        
    else:
        get_ls_similar_labels_label_weighted(pickle_data, lbl, k, False)
    
def get_simlarlabel_byimg_ls():
    print("identifies and lists k most relevant images, along with their scores, under the selected latent space.")

def task10(label,latentk,feature_model,dimred,latsem,k,odd_feature_collection,feature_collection,similarity_collection,caltech101):
    mod_path = Path(__file__).parent.parent
    pkl_file_path = str(mod_path)+"/LatentSemantics/"
    
    if feature_model == "Color Moments":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_color_moments_"+str(latentk)+"_output.pkl"
    

    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_hog_"+str(latentk)+"_output.pkl"    
        

    elif feature_model == "ResNet-AvgPool-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_avgpool_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_avgpool_"+str(latentk)+"_output.pkl"
        

    elif feature_model == "ResNet-Layer3-1024":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_layer3_"+str(latentk)+"_output.pkl"
       
    elif feature_model == "ResNet-FC-1000":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_fc_"+str(latentk)+"_output.pkl"            

    elif feature_model == "RESNET":
        if dimred!="":
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+dimred+"_"+str(latentk)+"_output.pkl"
        else:
            pkl_file_path += "latent_semantics_"+latsem[2]+"_resnet_"+str(latentk)+"_output.pkl"
    
    print(pkl_file_path)

    with open(pkl_file_path,'rb') as file:
        print('File path is '+pkl_file_path)
        __,pickle_data = pickle.load(file)
        
    mat_file_path = str(mod_path)+"/LatentSemantics/"

    try:

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')
        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(mat_file_path,feature_collection)

        data = scipy.io.loadmat(mat_file_path+'arrays.mat')

        labels = data['labels']
        cm_features = data['cm_features']
        hog_features = data['hog_features']
        avgpool_features = data['avgpool_features']
        layer3_features = data['layer3_features']
        fc_features = data['fc_features']
        resnet_features = data['resnet_features']
    
    print('Pickle File Loaded')
        
    print(pickle_data.shape)

    print(labels.shape,labels)

    
    if pickle_data.shape[0] == 4339:

        image_similarities,_ = get_ls_similar_images_from_label_image_weighted(pickle_data,label, k, feature_collection)

        final_scores = dict(sorted(image_similarities.items(), key=lambda item: item[1], reverse=True)[:k])

        print(final_scores)
    
        #Format final output to call display method
        
        display_images_list=[]
        display_indices=[]
        display_similarity_scores=[]
        
        for key in final_scores.keys():

            imagedata = feature_collection.find_one({'_id':key})

            image = np.array(imagedata['image'], dtype=np.uint8)

            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            display_images_list.append(image)
            display_indices.append(key)
            display_similarity_scores.append(final_scores[key])
            
        #print("The lengths are :")
        #print(str(len(display_images_list))+" "+str(len(display_indices))+" "+str(len(display_similarity_scores)))
        
        #Call display method for final output
        
        display_images(display_images_list,display_indices,display_similarity_scores,0,0,"Similarity Score : ")


    elif pickle_data.shape[0] == 101:

        sim_la = get_ls_similar_labels_label_weighted(pickle_data, label, 1, True)

        print(get_class_name(list(sim_la.keys())[0]))

        matching_label = list(sim_la.keys())[0]

        similarity_calculator_by_label(matching_label,feature_model,k,odd_feature_collection,feature_collection,similarity_collection,caltech101)


###################################################################   PHASE 3 CODE  ###########################################################################

def euclidean(point, data):
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data)**2, axis=1))
class KMeans:

    def __init__(self, n_clusters=8, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            print("Iteration "+str(iteration))
            for idx in tqdm(range(len(X_train))):
                x = X_train[idx]
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        print("Classifying")
        for idx in tqdm(range(len(X))):
            x = X[idx]
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

    

def classifier(cltype,feature_collection,odd_feature_collection,k=0):

    mod_path = Path(__file__).parent.parent
    mat_file_path = mod_path.joinpath("LatentSemantics","")
    mat_file_path = str(f'{mat_file_path}{os.sep}')
    even_desc_path = mod_path.joinpath("LatentSemantics","arrays.mat")
    odd_desc_path = mod_path.joinpath("LatentSemantics","arrays_odd.mat")

    #print(mat_file_path,even_desc_path,odd_desc_path)

    try:

        data = scipy.io.loadmat(str(even_desc_path))
        labels = data['labels']
        layer3_features = data['layer3_features']

        odd_data = scipy.io.loadmat(str(odd_desc_path))
        odd_labels = odd_data['labels']
        odd_layer3_features = odd_data['layer3_features']

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(str(mat_file_path),feature_collection)
        store_by_feature_odd(str(mat_file_path),odd_feature_collection)

        data = scipy.io.loadmat(str(even_desc_path))
        labels = data['labels']
        layer3_features = data['layer3_features']

        odd_data = scipy.io.loadmat(str(odd_desc_path))
        odd_labels = odd_data['labels']
        odd_layer3_features = odd_data['layer3_features']
        
    if cltype == "Nearest Neighbors":

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(layer3_features)
        # View results
        class_centers, classification = kmeans.evaluate(layer3_features)
        centroid_dict = {}
        centroid_label_vote = {}
        centroid_label_mapping = {}

        print("Taking Label Vote")
        for idx in tqdm(range(len(class_centers))):
            cluster_id = int(classification[idx])

            if cluster_id not in centroid_dict.keys():
                centroid_dict[cluster_id] = class_centers[idx]

            label = np.where(labels[idx]==1)[0][0]

            if cluster_id in centroid_label_vote.keys():
                if label in centroid_label_vote[cluster_id]:
                    centroid_label_vote[cluster_id][label]+=1
                else:
                    centroid_label_vote[cluster_id][label]=1
            else:
                centroid_label_vote[cluster_id] = {label: 1}
        
        for key in centroid_label_vote.keys():
            centroid_label_mapping[key] = max(centroid_label_vote[key],key = centroid_label_vote[key].get)

        centroid_label_mapping = {key:value for key, value in sorted(centroid_label_mapping.items(), key=lambda item: int(item[0]))}
        print("Centroid to label mapping:")
        print(centroid_label_mapping)

        print("Even features")
        print(type(layer3_features),layer3_features.shape)

        print("Odd features")
        print(type(odd_layer3_features),odd_layer3_features.shape)

        class_centers,classification = kmeans.evaluate(odd_layer3_features)

        true_labels = []
        
        for idx in range(len(odd_labels)):
            true_labels.append(np.where(labels[idx]==1)[0][0])

        predictions = []

        confusion_matrix = np.zeros((101,101))
        print(confusion_matrix.shape)

        for idx in range(len(classification)):
            c = int(classification[idx])
            predictions.append(centroid_label_mapping[int(c)])

            l = int(true_labels[idx])
            confusion_matrix[l][c]+=1

        labelwise_metrics = {}
        for idx in range(101):
            tp = confusion_matrix[idx][idx]
            fn = sum(confusion_matrix[idx]) - confusion_matrix[idx][idx]
            tn = 0
            fp = 0
            for r in range(101):
                for c in range(101):
                    if r!= idx and c!=idx:
                        tn+=1
                    if c == idx and r!=idx:
                        fp+=1
            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            f1_score = (2*precision*recall) / (precision+recall)
            labelwise_metrics[idx] = {"Precision":precision,"Recall":recall,"F1-Score":f1_score}

        truecount = 0
        for idx in range(len(predictions)):
            if predictions[idx] == true_labels[idx]:
                truecount+=1

        accuracy = truecount/len(predictions)

        st.write("Accuracy Scores:")
        st.write("Overall Accuracy: "+str(accuracy))
        st.write(confusion_matrix)

        """with st.container():
            for idx in range(101):
                with st.expander("Label "+str(idx)):
                    st.write("Precision: "+str(precision))
                    st.write("Recall: "+str(recall))
                    st.write("F1-Score: "+str(f1_score))"""

    elif cltype == "Decision Tree":
        pass
    elif cltype == "PPR":
        pass
    
    
def ls_even_by_label():
    mod_path = Path(__file__).parent.parent
    ls_file_path = str(mod_path)+"/LatentSemantics/"
    try:

        data_even = scipy.io.loadmat(ls_file_path+'arrays.mat')
        data_odd  = scipy.io.loadmat(ls_file_path+'arrays_odd.mat')
        
        labels_even = data_even['labels']
        labels_odd = data_odd['labels']

        #Latent Semantic chosen is ResNet as Feature Model, K-Means as Dimensionality Reduction Technique and 'k' value as 5.
        pkl_file_path = ls_file_path+"latent_semantics_3_resnet_k-Means_5_output.pkl"
        with open(pkl_file_path,'rb') as file:
            print('File path is '+pkl_file_path)
            __,pickle_data = pickle.load(file)
        

            
        
            
            
    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print("Exception in ls_even_by_label "+e)
    
    




dataset_size = 8677
dataset_mean_values = [0.5021372281891864, 0.5287581550675707, 0.5458470856851454]
dataset_std_dev_values = [0.24773670511666424, 0.24607509728422117, 0.24912913964278197]

data = {
	0: "Faces", 1: "Faces_easy", 2: "Leopards", 3: "Motorbikes", 4: "accordion", 5: "airplanes", 6: "anchor", 7: "ant", 8: "barrel", 9: "bass",
	10: "beaver", 11: "binocular", 12: "bonsai", 13: "brain", 14: "brontosaurus", 15: "buddha", 16: "butterfly", 17: "camera", 18: "cannon", 19: "car_side",
	20: "ceiling_fan", 21: "cellphone", 22: "chair", 23: "chandelier", 24: "cougar_body", 25: "cougar_face", 26: "crab", 27: "crayfish", 28: "crocodile", 29: "crocodile_head",
	30: "cup", 31: "dalmatian", 32: "dollar_bill", 33: "dolphin", 34: "dragonfly", 35: "electric_guitar", 36: "elephant", 37: "emu", 38: "euphonium", 39: "ewer",
	40: "ferry", 41: "flamingo", 42: "flamingo_head", 43: "garfield", 44: "gerenuk", 45: "gramophone", 46: "grand_piano", 47: "hawksbill", 48: "headphone", 49: "hedgehog",
	50: "helicopter", 51: "ibis", 52: "inline_skate", 53: "joshua_tree", 54: "kangaroo", 55: "ketch", 56: "lamp", 57: "laptop", 58: "llama", 59: "lobster",
	60: "lotus", 61: "mandolin", 62: "mayfly", 63: "menorah", 64: "metronome", 65: "minaret", 66: "nautilus", 67: "octopus", 68: "okapi", 69: "pagoda",
	70: "panda", 71: "pigeon", 72: "pizza", 73: "platypus", 74: "pyramid", 75: "revolver", 76: "rhino", 77: "rooster", 78: "saxophone", 79: "schooner",
	80: "scissors", 81: "scorpion", 82: "seahorse", 83: "snoopy", 84: "soccer_ball", 85: "stapler", 86: "starfish", 87: "stegosaurus", 88: "stop_sign", 89: "strawberry",
	90: "sunflower", 91: "tick", 92: "trilobite", 93: "umbrella", 94: "watch", 95: "water_lilly", 96: "wheelchair", 97: "wild_cat", 98: "windsor_chair", 99: "wrench", 100: "yin_yang"}