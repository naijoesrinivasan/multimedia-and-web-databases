import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from scipy.stats import skew
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing as p
import pickle
import scipy.io
import tensorly as tl
import scipy.misc
import tensortools as tt
from tensortools.operations import unfold as tt_unfold, khatri_rao
import os
from scipy import linalg
from FeatureDescriptors.SimilarityScoreUtils import *
from Utilities.DisplayUtils import *
import streamlit as st
from pathlib import Path
from heapq import nsmallest
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.manifold import MDS
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

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

    return distances, centroids

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
    top_k_indices = np.argsort(latent_semantics.sum(axis=0))[::-1][:k]
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
def get_index_for_label(label, dataset,baseIndex):
    index = []
    for i in range(baseIndex, len(dataset.y), 2):
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

def get_labels_similarity_matrix(feature_model, odd_feature_collection, feature_collection, similarity_collection, dataset,baseIndex):
    labels = [label for label in range(101)]
    
    label_sim_matrix = np.nan * np.zeros((101,101))
    print(label_sim_matrix.shape)

    for idx in range(101):
        label_sim_matrix[idx][idx] = 1

    for labelx in tqdm(labels):
        labelx_idx = get_index_for_label(labelx, dataset,baseIndex)

        for labely in labels:
            if labelx == labely: continue
            if np.isnan(label_sim_matrix[labelx][labely]):

                labely_idx = get_index_for_label(labely, dataset,baseIndex)
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

        fetchedarray = doc['fc_softmax_descriptor']

        resnetarray = [0 if pd.isna(x) else x for x in fetchedarray]

        resnetarray = np.array(resnetarray)

        resnet_features.append(resnetarray)

    scipy.io.savemat(output_file+'arrays_odd.mat', {'labels': labels, 'cm_features': cm_features, 'hog_features':hog_features, 'avgpool_features': avgpool_features,'layer3_features':layer3_features, 'fc_features': fc_features,'resnet_features' : resnet_features})  

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

class kNN:

    def __init__(self, collection, labels,k = 10):
        self.k = k
        self.collection = collection
        self.labels = labels

    def predict(self):

        classifications = []

        for idx in tqdm(range(1,8677,2)):

            scores = self.collection.find_one({'_id':idx})['avgpool_descriptor']

            even_scores = {}

            for imgid in scores.keys():
                if int(imgid)%2 == 0:
                    even_scores[int(imgid)] = scores[imgid]

            #print("Scores present for "+str(len(scores.keys()))+" images")

            even_scores = dict(sorted(even_scores.items(), key = lambda x: x[1])[-self.k:])

            top_k_even_indices = list(even_scores.keys())

            #print("Most similar images: "+str(top_k_even_indices))

            label_votes = {}
            for even_index in top_k_even_indices:

                l = np.where(self.labels[int(even_index/2)]==1)[0][0]

                if l in label_votes.keys():
                    label_votes[l]+=1
                else:
                    label_votes[l]=1

            prediction = max(label_votes, key=label_votes.get)
            #print("Image ID: "+str(idx)+" Label:"+str(prediction))

            classifications.append(prediction)
        
        return classifications
    
def pagerank(adj_matrix, labels, personalization, teleport_prob=0.15, max_iter=100):

    #Initialize pagerank vector
    pagerank = (1-teleport_prob)*np.dot(adj_matrix,personalization) + teleport_prob*(np.ones(adj_matrix.shape[0])/adj_matrix.shape[0])
    pagerank/=np.sum(pagerank)

    for _ in tqdm(range(max_iter)):

        label_totals = {}

        for index in range(adj_matrix.shape[0]):
            if labels[index] in label_totals.keys():
                label_totals[labels[index]]+=1
            else:
               label_totals[labels[index]]=1 

        reseed = []

        for index in range(adj_matrix.shape[0]):
            reseed.append(pagerank[index]/label_totals[labels[index]])

        reseed = np.array(reseed)
        reseed/=np.sum(reseed)

        #calculate PPR-G score
        pagerank = (1-teleport_prob)*np.dot(adj_matrix,pagerank) + teleport_prob * reseed
        pagerank/=np.sum(pagerank)

    return pagerank

class PersonalizedPageRankClassifier:
    def __init__(self, teleport_prob=0.15, max_iter=100, tol=1e-5):
        self.teleport_prob = teleport_prob
        self.max_iter = max_iter
        self.tol = tol
        self.classifiers = []

    def fit(self, X_train, y_train, personalization = None):
        self.classes_ = np.unique(y_train)
        self.personalization = personalization

        # Compute personalized PageRank for the entire training set
        pagerank_vector = pagerank(X_train, y_train, personalization, teleport_prob=self.teleport_prob, max_iter=self.max_iter)
        
        #pagerank_vector = pagerank_vector/np.sum(pagerank_vector)
        
        # Store the computed pagerank_vector
        self.pagerank_vector = pagerank_vector

        with st.expander("PageRank Values for Training Data"):
            st.write(list(self.pagerank_vector))
        with st.expander("Pagerank Values Analysis"):
            st.write("Min Value: "+str(np.min(self.pagerank_vector)))
            st.write("Max Value: "+str(np.max(self.pagerank_vector)))
            st.write("Mean Value: "+str(np.mean(self.pagerank_vector)))

    def predict(self, X_test, train_labels):
        predictions = []

        for idx in range(X_test.shape[0]):
            pred_pagerank = (1-self.teleport_prob)*X_test[idx]*self.pagerank_vector + self.teleport_prob*X_test[idx]
            pred_pagerank/=np.sum(pred_pagerank)
            predictions.append(train_labels[np.argmax(pred_pagerank)])

        return predictions
    
class DecisionTree:
    def __init__(self, max_depth=None, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y, search_size = 10,depth=0):
        # Store training data and labels in the node
        self.X = X
        self.y = y

        print("Processing depth: "+str(depth))

        # Check if the node should be a leaf
        if self.should_stop_splitting(y, depth):
            self.label = self.most_common_label(y)
            return

        # Find the best split
        feature_index, threshold = self.find_best_split(X, y,search_size)

        # If no split is found, make it a leaf node
        if feature_index is None:
            self.label = self.most_common_label(y)
            return

        # Split the data and recursively build the tree
        mask = X[:, feature_index] <= threshold
        self.left = DecisionTree(self.max_depth, self.min_size)
        self.left.fit(X[mask], y[mask], search_size=search_size,depth = depth + 1)

        self.right = DecisionTree(self.max_depth, self.min_size)
        self.right.fit(X[~mask], y[~mask], search_size=search_size,depth = depth + 1)

        self.feature_index = feature_index
        self.threshold = threshold

    def should_stop_splitting(self, y, depth):
        # Check if the node should be a leaf based on hyperparameters
        return (self.max_depth is not None and depth == self.max_depth) or len(set(y)) == 1 or len(y) <= self.min_size

    def find_best_split(self, X, y,search_size):
        # Find the best split based on Gini impurity
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        rand_indexes = np.random.choice(range(X.shape[1]),size = search_size, replace=False)

        for feature_index in tqdm(rand_indexes):
            thresholds = sorted(set(X[:, feature_index]))

            for threshold in thresholds:
                mask = X[:, feature_index] <= threshold
                gini = self.calculate_gini_impurity(y, mask)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def calculate_gini_impurity(self, y, mask):
        # Calculate Gini impurity for a split
        size = len(y)
        if size == 0:
            return 0

        p_left = len(y[mask]) / size
        p_right = len(y[~mask]) / size

        gini_left = 1 - sum((np.sum(y[mask] == c) / len(y[mask])) ** 2 for c in set(y[mask]))
        gini_right = 1 - sum((np.sum(y[~mask] == c) / len(y[~mask])) ** 2 for c in set(y[~mask]))

        gini = p_left * gini_left + p_right * gini_right
        return gini

    def most_common_label(self, y):
        # Return the most common label in the node for a NumPy ndarray
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        most_common_label = unique_labels[most_common_index]
        return most_common_label

    def predict_single(self, sample):
        # Predict the label for a single sample
        if hasattr(self, 'label'):
            return self.label

        if sample[self.feature_index] <= self.threshold:
            return self.left.predict_single(sample)
        else:
            return self.right.predict_single(sample)

    def predict(self, X):
        # Predict labels for multiple samples
        return [self.predict_single(sample) for sample in X]
    
    def save_model(self, filename):
        # Save the decision tree model to a file using joblib
        joblib.dump(self, filename)

def load_model(filename):
    # Load a saved decision tree model from a file using joblib
    return joblib.load(filename)
                
def display_scores(confusion_matrix,true_labels,predictions,acc=None):

    with st.expander("Confusion Matrix for Classification",expanded = True):
        conf = pd.DataFrame(data=confusion_matrix, columns=[str(i) for i in range(0, confusion_matrix.shape[1])])
        st.dataframe(conf,width = 200000)

    if acc!=None:
        labelslist = []
        for label in true_labels:
            labelslist.append(get_class_name(label))
        table = {
        'Image Index': list(range(1,8677,2)),
        'True Labels': labelslist,
        'Predicted Labels': predictions
        }
    else:
        table = {
        'Image Index': list(range(1,8677,2)),
        'True Labels': true_labels,
        'Predicted Labels': predictions
        }

    # Create a DataFrame
    table = pd.DataFrame(table)

    with st.expander("Predicted Labels",expanded=True):
        st.dataframe(table,width = 200000)

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
        if precision == 0 or recall == 0:
            f1_score = 0
        else:
            f1_score = (2*precision*recall) / (precision+recall)
        labelwise_metrics[idx] = {"Precision":precision,"Recall":recall,"F1-Score":f1_score}

    truecount = 0
    for idx in range(len(predictions)):
        if predictions[idx] == true_labels[idx]:
            truecount+=1
 
    accuracy = truecount/len(predictions)
    if acc!=None:
        accuracy = acc

    st.write("Accuracy Scores:")
    st.write("Overall Accuracy: "+str(accuracy))
    #st.write(confusion_matrix)

    df = pd.DataFrame({label: values for label, values in labelwise_metrics.items()}).T
    st.dataframe(df)

    with st.container():
        for idx in range(101):
            with st.expander("Label "+str(idx),expanded = True):
                st.write("Precision: "+str(labelwise_metrics[idx]['Precision']))
                st.write("Recall: "+str(labelwise_metrics[idx]['Recall']))
                st.write("F1-Score: "+str(labelwise_metrics[idx]['F1-Score']))


def classifier(img_id,cltype,feature_collection,odd_feature_collection,similarity_collection,dataset,k=0,teleport_prob=0):

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

    if cltype == "Decision Tree":

        max_depth = 10
        min_size = 25
        search_size = 25

        tree = DecisionTree(max_depth=max_depth,min_size=min_size)

        even_labels = []
        
        for idx in range(len(labels)):
            even_labels.append(np.where(labels[idx]==1)[0][0])

        scaler = StandardScaler()
        layer3_features_scaled = scaler.fit_transform(layer3_features)
        treepath = mod_path.joinpath("Classifiers","DecisionTree",str(max_depth)+"_"+str(min_size)+"_"+str(search_size)+".joblib")

        if os.path.exists(treepath):
            tree = load_model(treepath)

        else:
            tree.fit(layer3_features_scaled,np.array(even_labels),search_size)
            tree.save_model(treepath)

        true_labels = []
        
        for idx in range(len(odd_labels)):
            true_labels.append(np.where(odd_labels[idx]==1)[0][0])

        odd_layer3_features_scaled = scaler.fit_transform(odd_layer3_features)
        predictions = tree.predict(odd_layer3_features_scaled)

        confusion_matrix = np.zeros((101,101))

        for idx in range(len(predictions)):

            l = int(true_labels[idx])
            c = predictions[idx]
            confusion_matrix[l][c]+=1

        if img_id%2==0:

            document = feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text ="Query input not an odd index"
        
        else:

            document = odd_feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text = "Query Image Tagged to Label "+str(predictions[img_id//2])+" "+get_class_name(predictions[img_id//2])

        with st.expander("Query Image and Label"):

            display_image_centered(np.array(image),str(img_id))

            st.markdown(markdown_text)

        display_scores(confusion_matrix,true_labels,predictions)

    elif cltype == "Nearest Neighbors":

        even_labels = []
        
        for idx in range(len(labels)):
            even_labels.append(np.where(labels[idx]==1)[0][0])

        nnclassifier = kNN(similarity_collection,labels,k = k)

        true_labels = []
        
        for idx in range(len(odd_labels)):
            true_labels.append(np.where(odd_labels[idx]==1)[0][0])

        predictions = nnclassifier.predict()

        confusion_matrix = np.zeros((101,101))

        for idx in range(len(predictions)):

            l = int(true_labels[idx])
            c = predictions[idx]
            confusion_matrix[l][c]+=1

        if img_id%2==0:

            document = feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text ="Query input not an odd index"
        
        else:

            document = odd_feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text = "Query Image Tagged to Label "+str(predictions[img_id//2])+" "+get_class_name(predictions[img_id//2])

        with st.expander("Query Image and Label"):

            display_image_centered(np.array(image),str(img_id))

            st.markdown(markdown_text)

        display_scores(confusion_matrix,true_labels,predictions)

    elif cltype == "PPR":

        even_labels = []
        
        for idx in range(len(labels)):
            even_labels.append(np.where(labels[idx]==1)[0][0])

        print("Using Teleport Prob"+str(teleport_prob))
        ppr = PersonalizedPageRankClassifier(teleport_prob=teleport_prob)

        adj_matrix = []

        personalization = []
        label_totals = {}

        adj_matfile_path = str(mod_path.joinpath("Classifiers","PPR","Pagerank_Adjacencies"+str(teleport_prob)+".mat"))

        if os.path.exists(adj_matfile_path):

            ppr_data = scipy.io.loadmat(adj_matfile_path)

            adj_matrix = ppr_data['adj_matrix']

            odd_adj_matrix = ppr_data['odd_adj_matrix']

            odd_labels = ppr_data['odd_labels']

            ppr.pagerank_vector = ppr_data['pagerank']

        else:

            print("Building Adjacency Matrix")
            for idx in tqdm(range(0,8677,2)):

                scores = similarity_collection.find_one({'_id':idx})['avgpool_descriptor']

                label = even_labels[int(idx/2)]
                even_scores = {}
                label_total = 0
                label_count = 0
                
                for imgid in scores.keys():
                    if int(imgid)%2 == 0:
                        even_scores[int(imgid)] = scores[imgid]
                        if even_labels[int(int(imgid)/2)] == label:
                            label_total += scores[imgid]
                            label_count += 1

                personalization.append(label_total/label_count)

                if label in label_totals.keys():
                    label_totals[label]+=label_total/label_count
                else:
                    label_totals[label]=label_total/label_count

                #print("Scores present for "+str(len(scores.keys()))+" images")

                even_scores = dict(sorted(even_scores.items(), key = lambda x: x[1])[-5:])

                top_k_even_indices = list(even_scores.keys())

                row = np.zeros(4339)

                for indice in top_k_even_indices:
                    row[int(indice/2)] = 1

                adj_matrix.append(row)

            for idx in range(len(personalization)):
                personalization[idx]/=label_totals[even_labels[idx]]

            adj_matrix = np.array(adj_matrix)
            even_labels = np.array(even_labels)
            print("Adjacency Matrix: "+str(adj_matrix.shape)+" Labels: "+str(even_labels.shape))

            ppr.fit(adj_matrix,even_labels,personalization=personalization)

            odd_adj_matrix = []

            print("Building Adjacency Matrix")
            for idx in tqdm(range(1,8677,2)):

                scores = similarity_collection.find_one({'_id':idx})['avgpool_descriptor']

                even_scores = {}

                for imgid in scores.keys():
                    if int(imgid)%2 == 0:
                        even_scores[int(imgid)] = scores[imgid]

                #print("Scores present for "+str(len(scores.keys()))+" images")

                even_scores = dict(sorted(even_scores.items(), key = lambda x: x[1])[-10:])

                top_k_even_indices = list(even_scores.keys())

                row = np.zeros(4339)

                for indice in top_k_even_indices:
                    row[int(indice/2)] = 1

                odd_adj_matrix.append(row)

            odd_adj_matrix = np.array(odd_adj_matrix)

            scipy.io.savemat(adj_matfile_path, {'adj_matrix': adj_matrix, 'odd_adj_matrix': odd_adj_matrix, 'even_labels':even_labels, 'odd_labels': odd_labels, 'pagerank': ppr.pagerank_vector})

        odd_adj_matrix = np.array(odd_adj_matrix)

        predictions = ppr.predict(odd_adj_matrix,even_labels)

        confusion_matrix = np.zeros((101,101))

        true_labels = []
        
        for idx in range(len(odd_labels)):
            true_labels.append(np.where(odd_labels[idx]==1)[0][0])

        for idx in range(len(predictions)):

            l = int(true_labels[idx])
            c = predictions[idx]
            confusion_matrix[l][c]+=1

        if img_id%2==0:

            document = feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text ="Query input not an odd index"
        
        else:

            document = odd_feature_collection.find_one({'_id':img_id})
            image = np.array(document['image'], dtype=np.uint8)
            markdown_text = "Query Image Tagged to Label "+str(predictions[img_id//2])+" "+get_class_name(predictions[img_id//2])

        with st.expander("Query Image and Label"):

            display_image_centered(np.array(image),str(img_id))

            st.markdown(markdown_text)

        display_scores(confusion_matrix,true_labels,predictions)


####################################################### Task 1 Methods ##############################################################################
    
    
def calculate_label_from_semantic(even_label_weighted_latent_semantics,odd_latent_semantics):
    print('Enter calculate_label_from_semantic')
    output_labels=[]
    
   # Compute Distances for every row from the odd image latent semantics with every row (label) from the label weighted semantics (even images)
    # for idx in tqdm(range(0,odd_latent_semantics.shape[0])):
    #     sim_scores=[]
    #     for cmpidx in range(0,even_label_weighted_latent_semantics.shape[0]):
    #         #sim_scores.append(euclidean_distance_calculator(odd_latent_semantics[idx],even_label_weighted_latent_semantics[cmpidx]))
    #         sim_scores.append(cosine_similarity_calculator(odd_latent_semantics[idx],even_label_weighted_latent_semantics[cmpidx]))

    #         #print(min(sim_scores),max(sim_scores))
    #     output_labels.append(np.argmin(sim_scores))
        
    # even_label_weighted_latent_semantics_transpose = np.array(even_label_weighted_latent_semantics).T

    image_label_latent_semantic = np.dot(odd_latent_semantics,even_label_weighted_latent_semantics)
    print('Dot product complete')
    print(image_label_latent_semantic.shape)
    #print(image_label_latent_semantic) # 4338,5
    for idx in tqdm(range(0,image_label_latent_semantic.shape[0])):
        scores=[]
        for cmpidx in range(0,even_label_weighted_latent_semantics.shape[0]):
            scores.append(cosine_similarity_calculator(image_label_latent_semantic[idx],even_label_weighted_latent_semantics[cmpidx]))
        output_labels.append(np.argmax(scores))
            

    print(output_labels)
    print('Exit calculate_label_from_semantic')
    return output_labels


def generate_label_weighted_semantics(image_data,feature_space,k,feature_collection, odd_feature_collection, similarity_collection):

    representation_image_index_by_label = []

    required_resnet_features = []

    
    for label in tqdm(range(0,101)):
        representation_image_index_by_label.append(similarity_calculator_by_label(label,feature_space,1,odd_feature_collection,feature_collection,similarity_collection,Caltech101))

    
    for idx in representation_image_index_by_label:
        required_resnet_features.append(image_data['resnet_features'][int(idx/2)])
    
    #print(required_resnet_features.shape)

    label_weighted_latent_semantics, centroids = kmeans_decomposition(np.array(required_resnet_features),k)
    
    return label_weighted_latent_semantics, centroids
        


def get_image_label_latent_semantic(similarity_collection,data_odd,dataset):
    odd_labels = data_odd['labels']
    image_label_matrix=[]
    

    for i in tqdm(range(1,len(dataset),2)):
        image_label_weights =[0]*101
        label_wise_scores={}
        similarity_scores = similarity_collection.find_one({'_id': i})
        for j in range(1,len(dataset),2):
            img_label = np.where(odd_labels[(int)(j/2)]==1)[0][0]
            if img_label not in label_wise_scores:
                label_wise_scores[img_label]=[]
            label_wise_scores[img_label].append(similarity_scores['fc_softmax_descriptor'][str(j)])
            for key in label_wise_scores.keys():
                scores = label_wise_scores[key]
                image_label_weights[key] = sum(scores)/len(scores)
        image_label_matrix.append(image_label_weights)
    
    return np.array(image_label_matrix)



def ls_even_by_label(k,feature_collection, odd_feature_collection, similarity_collection,dataset):
    mod_path = Path(__file__).parent.parent
    ls_file_path = str(mod_path)+"/LatentSemantics/"
    k=5
    try:
        data_even = scipy.io.loadmat(ls_file_path+'arrays.mat')
        data_odd  = scipy.io.loadmat(ls_file_path+'arrays_odd.mat')

        print('Descriptor Mat Files Loaded Successfully')
    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print("Exception in ls_even_by_label "+str(e))
        store_by_feature(str(ls_file_path),feature_collection)
        store_by_feature_odd(str(ls_file_path),odd_feature_collection)
        data_even = scipy.io.loadmat(ls_file_path+'arrays.mat')
        data_odd  = scipy.io.loadmat(ls_file_path+'arrays_odd.mat')
        
        #labels_even = data_even['labels']
        #labels_odd = data_odd['labels']

    #Latent Semantic chosen is ResNet as Feature Model, K-Means as Dimensionality Reduction Technique and 'k' value as 5 for the even images in the dataset. 
    try:
        pkl_file_path = ls_file_path+"latent_semantics_3_resnet_k-Means_"+str(k)+"_output.pkl"  #Change this file path after new pickle file has been created. 
        with open(pkl_file_path,'rb') as file:
            print('File path is '+pkl_file_path)
            _,even_label_weighted_latent_semantics = pickle.load(file)
            print('Even LS Pickle File Loaded')
        # pkl_file_path = ls_file_path+"TASK5_fc_k5_kmeans.csv"  #Change this file path after new pickle file has been created. 
        # print('File path is '+pkl_file_path)
        # even_label_weighted_latent_semantics = pd.read_csv(pkl_file_path)

        print('Even LS Pickle File Loaded')

    except (FileNotFoundError) as e:
        even_label_weighted_latent_semantics, centroids = generate_label_weighted_semantics(data_even,'RESNET',k,feature_collection, odd_feature_collection, similarity_collection)
        pickle.dump((even_label_weighted_latent_semantics,centroids), open(pkl_file_path, 'wb+'))
        print('Pickle File Created : '+pkl_file_path)
    
    print(even_label_weighted_latent_semantics.shape)
    

    #Calculate Latent Semantics for Odd Images
    try:
        odd_ls_file_path = ls_file_path+"odd_latent_semantics_"+str(k)+".pkl"
        with open(odd_ls_file_path,'rb') as file:
            print('File path is '+odd_ls_file_path)
            odd_latent_semantics = pickle.load(file)
            print('Odd Latent Semantic Pickle File Loaded')
            print(odd_latent_semantics.shape)

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print('Calculating Latent Semantics for Odd Images')
        odd_resnet_features = data_odd['resnet_features']
        print(odd_resnet_features.shape)
        odd_latent_semantics = get_image_label_latent_semantic(similarity_collection,data_odd,dataset)
        
        """odd_latent_semantics = []
        print("Calculating odd LS")
        for idx in tqdm(range(odd_resnet_features.shape[0])):
            feature = odd_resnet_features[idx]
            semantic = []
            for centroid in centroids:
                semantic.append(euclidean_distance_calculator(feature,centroid))
            odd_latent_semantics.append(semantic)

        odd_latent_semantics = np.array(odd_latent_semantics)"""

        #odd_latent_semantics = reduce_dimensionality(odd_resnet_features, k, "LDA")

        print('Odd Latent Semantics Calculated')
        print(odd_latent_semantics.shape)
        pickle.dump(odd_latent_semantics, open(odd_ls_file_path, 'wb+'))
        print('Pickle File Created : '+odd_ls_file_path)

    
    true_labels = []
    odd_labels = data_odd['labels']
        
    for idx in range(len(odd_labels)):
        true_labels.append(np.where(odd_labels[idx]==1)[0][0])

    predictions = calculate_label_from_semantic(even_label_weighted_latent_semantics,odd_latent_semantics)

    print('Predictions length is '+str(len(predictions)))

    confusion_matrix = np.zeros((101,101))
    

    for idx in range(len(predictions)):
        c = int(predictions[idx])
        l = int(true_labels[idx])
        confusion_matrix[l][c]+=1

    display_scores(confusion_matrix,true_labels,predictions)
    
class LSH:
    def __init__(self, num_layers, num_hashes, input_dim):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.input_dim = input_dim
        self.hash_tables = [{} for _ in range(num_layers)]
        self.random_planes = [np.random.randn(num_hashes, input_dim) for _ in range(num_layers)]

    def hash_vector(self, vector, planes):
        return ''.join(['1' if np.dot(vector, plane) > 0 else '0' for plane in planes])

    def index_vector(self, vector, index):
        for i in range(self.num_layers):
            hash_value = self.hash_vector(vector, self.random_planes[i])
            hash_key = tuple(vector)  # Convert NumPy array to tuple
            if hash_value not in self.hash_tables[i]:
                self.hash_tables[i][hash_value] = []
            self.hash_tables[i][hash_value].append((hash_key, index))

    def index_vectors(self, vectors):
        for i, vector in enumerate(vectors):
            self.index_vector(vector, i)

    def hash_query_vector(self, query_vector):
        hashed_buckets = []
        for i in range(self.num_layers):
            hash_value = self.hash_vector(query_vector, self.random_planes[i])
            if hash_value in self.hash_tables[i]:
                bucket = self.hash_tables[i][hash_value]
                hashed_buckets.append((i+1, hash_value, bucket))
            else:
                hashed_buckets.append((i+1, hash_value, []))
        return hashed_buckets

    def calculate_distances(self, query_vector, hashed_buckets):
        distances = []
        for layer, _, bucket in hashed_buckets:
            layer_distances = []
            for item in bucket:
                vector = np.array(item[1])  # Retrieve the vector from the bucket
                distance = np.linalg.norm(query_vector - vector)  # Calculate Euclidean distance
                layer_distances.append((item[0], distance))  # Store index and distance
            distances.append((layer, layer_distances))
        return distances

    def display_buckets(self, layer):
        with st.expander("Layer "+str(layer)+" Buckets:"):
            if layer <= 0 or layer > self.num_layers:
                st.write("Layer index out of range.")
                return

            for key, value in sorted(self.hash_tables[layer - 1].items()):
                st.write("Hash Value: "+str(key)+" Items: "+str(len(value)))
                #for item in value:
                #    st.write("Index: "+str(item[1])+", Vector: Shape:"+str(len(item[0]))+ " Min:"+str(np.min(item[0]))+" Max: "+str(np.max(item[0])))

def lsh_calc(feature_collection,num_layers, num_hashes):

    mod_path = Path(__file__).parent.parent
    mat_file_path = mod_path.joinpath("LatentSemantics","")
    mat_file_path = str(f'{mat_file_path}{os.sep}')
    even_desc_path = mod_path.joinpath("LatentSemantics","arrays.mat")
    odd_desc_path = mod_path.joinpath("LatentSemantics","arrays_odd.mat")

    try:
        data_even = scipy.io.loadmat(even_desc_path)
        
        print('Descriptor Mat Files Loaded Successfully')
    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print("Exception in ls_even_by_label "+str(e))
        store_by_feature(str(mat_file_path),feature_collection)
        data_even = scipy.io.loadmat(even_desc_path)

    feature_desc_array = data_even['avgpool_features']
    labels = data_even['labels']
    even_labels = []
    image_ids = list(range(0,8677,2))
            
    for idx in range(len(labels)):
        even_labels.append(np.where(labels[idx]==1)[0][0])  

    lsh = LSH(num_layers, num_hashes, feature_desc_array.shape[1])
    lsh.index_vectors(feature_desc_array)

    lsh_path = str(mod_path.joinpath("LatentSemantics","lsh_"+str(num_layers)+"_"+str(num_hashes)+".pkl"))

    if os.path.exists(lsh_path):
        with open(lsh_path,'rb') as file:
            lsh = pickle.load(file)
    else:
        pickle.dump(lsh,open(lsh_path, 'wb+'))

    return lsh

def lsh_search(feature_collection,odd_feature_collection,lsh,query_image,t):

    mod_path = Path(__file__).parent.parent
    mat_file_path = mod_path.joinpath("LatentSemantics","")
    mat_file_path = str(f'{mat_file_path}{os.sep}')
    even_desc_path = mod_path.joinpath("LatentSemantics","arrays.mat")
    odd_desc_path = mod_path.joinpath("LatentSemantics","arrays_odd.mat")

    try:
        data_even = scipy.io.loadmat(even_desc_path)
        data_odd = scipy.io.loadmat(odd_desc_path)
        
        print('Descriptor Mat Files Loaded Successfully')
    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print("Exception in ls_even_by_label "+str(e))
        store_by_feature(str(mat_file_path),feature_collection)
        data_even = scipy.io.loadmat(even_desc_path)
        data_odd = scipy.io.loadmat(odd_desc_path)

    #lsh = lsh_calc(feature_collection,num_layers, num_hashes)

    if query_image%2==0:
        query = data_even['avgpool_features'][query_image//2]
    else:
        query = data_odd['avgpool_features'][query_image//2]

    hashed_buckets = lsh.hash_query_vector(query)

    unique_indices = set()
    hash_values = {}

    for layer, hash_value, bucket in hashed_buckets:
        for item in bucket:
            unique_indices.add(item[1]*2)  # Collecting unique indices
            hash_values[item[1]*2] = hash_value

    with st.expander("Unique Vector Indices:"):
        st.write("Number of Unique Indices considered:"+str(len(unique_indices)))
        st.write(unique_indices)

    distances = []  # To store calculated distances
    for _, _, bucket in hashed_buckets:
        for item in bucket:
            vector = np.array(item[0])  # Retrieve the vector from the bucket
            distance = np.linalg.norm(query - vector)  # Calculate Euclidean distance
            if (item[1],distance) not in distances:
                distances.append((item[1], distance))  # Store index and distance

    # Sort distances and retrieve k indices with smallest distances
    nearest_indices = [index*2 for index, _ in nsmallest(t, distances, key=lambda x: x[1])]

    if query_image%2==0:

        document = feature_collection.find_one({'_id':query_image})

    else:

        document = odd_feature_collection.find_one({'_id':query_image})

    image = np.array(document['image'], dtype=np.uint8)

    display_image_centered(np.array(image),str(query_image))

    show_ksimilar_list(nearest_indices,feature_collection,"")

    return nearest_indices, unique_indices, lsh, distances, hash_values

class LinearSVC:
    def __init__(self, C=1.0):
        self.C = C
        self.models = {}  # Dictionary to store models for each class

    def fit(self, X_train, y_train):
        unique_classes = np.unique(y_train)

        for class_label in unique_classes:
            binary_labels = np.where(y_train == class_label, 1, -1)

            m, n = X_train.shape
            P = matrix(np.outer(binary_labels, binary_labels) * np.dot(X_train, X_train.T))
            q = matrix(-np.ones(m))
            G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
            h = matrix(np.hstack([np.zeros(m), np.ones(m) * self.C]))
            A = matrix(binary_labels.reshape(1, -1),(1,m),'d')
            b = matrix(0.0)

            # Solve the quadratic optimization problem using CVXOPT
            solution = solvers.qp(P, q, G, h, A, b)

            # Extract Lagrange multipliers from the solution
            alpha = np.array(solution['x']).flatten()

            # Find support vectors
            support_vector_indices = (alpha > 1e-5)
            support_vectors = X_train[support_vector_indices]
            support_vector_labels = binary_labels[support_vector_indices]
            support_vector_alpha = alpha[support_vector_indices]

            # Save the model for the class
            self.models[class_label] = {
                'support_vectors': support_vectors,
                'support_vector_labels': support_vector_labels,
                'support_vector_alpha': support_vector_alpha
            }

    def predict(self, X_test):
        per_class_predictions = []

        for class_label, model in self.models.items():
            support_vectors = model['support_vectors']
            support_vector_labels = model['support_vector_labels']
            support_vector_alpha = model['support_vector_alpha']

            decision_function = np.dot(X_test, support_vectors.T)
            decision_function = np.sum(decision_function * support_vector_alpha * support_vector_labels, axis=1)

            per_class_predictions.append((class_label, decision_function))

        predictions = []

        #print(per_class_predictions[0][1].shape)
        for index in range(per_class_predictions[0][1].shape[0]):

            pred_vals = []

            for c in range(0,4):
                pred_vals.append(per_class_predictions[c][1][index])
            
            predictions.append(np.argmax(pred_vals))

        #st.write(predictions)

        return predictions
    
def relevance_feedback(t,option,query_image,feedback,distances,unique_indices,hash_values,feature_collection,odd_feature_collection,similarity_collection):

    mod_path = Path(__file__).parent.parent
    mat_file_path = mod_path.joinpath("LatentSemantics","")
    mat_file_path = str(f'{mat_file_path}{os.sep}')
    even_desc_path = mod_path.joinpath("LatentSemantics","arrays.mat")
    odd_desc_path = mod_path.joinpath("LatentSemantics","arrays_odd.mat")

    try:
        data_even = scipy.io.loadmat(even_desc_path)
        data_odd = scipy.io.loadmat(odd_desc_path)
        
        print('Descriptor Mat Files Loaded Successfully')
    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:
        print("Exception in ls_even_by_label "+str(e))
        store_by_feature(str(mat_file_path),feature_collection)
        data_even = scipy.io.loadmat(even_desc_path)
        data_odd = scipy.io.loadmat(odd_desc_path)

    X_train = []
    y_train = []
    X_test = []
    test_indices = {}

    for index in feedback.keys():
        X_train.append(data_even['avgpool_features'][index//2])
        y_train.append(feedback[index])

    #st.write(X_train)
    #st.write(y_train)

    for index in sorted(unique_indices):
        if index not in feedback.keys():
            X_test.append(data_even['avgpool_features'][index//2])
            test_indices[len(X_test)-1] = index
    
    if option == "SVM":

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create an SVM classifier with a linear kernel
        #svm_classifier = SVC(kernel='linear', C=0.01)
        svm_classifier = LinearSVC(C=0.00005)

        # Train the classifier on the training data
        svm_classifier.fit(X_train, y_train)

        y_train_pred = list(svm_classifier.predict(X_train))

        print("Train Accuracy: "+str(sum(1 for true, pred in zip(y_train, y_train_pred) if true == pred)/len(y_train)))

        # Make predictions on the test data
        y_pred = list(svm_classifier.predict(X_test))

        #df = pd.DataFrame({'Image ID': test_indices,'Relevance':y_pred})
        #st.dataframe(df)

        if query_image%2==0:

            document = feature_collection.find_one({'_id':query_image})

        else:

            document = odd_feature_collection.find_one({'_id':query_image})

        image = np.array(document['image'], dtype=np.uint8)

        fetchedarray = document['avgpool_descriptor']
                    
        avgpoolarray = [0 if pd.isna(x) else x for x in fetchedarray]

        avgpoolarray = np.array(avgpoolarray)

        display_image_centered(np.array(image),str(query_image))

        nearest_indices = []
        for index in feedback.keys():
            if feedback[index] == 3:
                if len(nearest_indices)<10:
                            nearest_indices.append(index)

        print(str(len(nearest_indices))+" Taken from feedback")

        relevant_indices = {}

        reorder_X_test = []

        for index in range(X_test.shape[0]):
            if y_pred[index] == 3:
                relevant_indices[len(reorder_X_test)] = test_indices[index]
                reorder_X_test.append(X_test[index])

        reorder_X_test = np.array(reorder_X_test)

        distances = [np.sqrt(np.sum((avgpoolarray - vector)**2)) for vector in reorder_X_test]

        required_num = t - len(nearest_indices)
        print("Required Num: "+str(required_num))
        min_indices = np.argsort(distances)[:required_num]

        for index in min_indices:

            nearest_indices.append(relevant_indices[index])

        print(str(len(nearest_indices))+" after score 3")

        if len(nearest_indices)<10:

            relevant_indices = {}

            reorder_X_test = []

            for index in range(X_test.shape[0]):
                if y_pred[index] == 2:
                    relevant_indices[len(reorder_X_test)] = test_indices[index]
                    reorder_X_test.append(X_test[index])

            reorder_X_test = np.array(reorder_X_test)

            distances = [np.sqrt(np.sum((avgpoolarray - vector)**2)) for vector in reorder_X_test]

            for index in feedback.keys():
                if feedback[index] == 2:
                    if len(nearest_indices)<10:
                                nearest_indices.append(index)

            required_num = t - len(nearest_indices)
            print("Required Num: "+str(required_num))
            min_indices = np.argsort(distances)[:required_num]

            for index in min_indices:

                nearest_indices.append(relevant_indices[index])

            print(str(len(nearest_indices))+" after score 2")

            if len(nearest_indices)<t:

                relevant_indices = {}

                reorder_X_test = []

                for index in range(X_test.shape[0]):
                    if y_pred[index] == 1:
                        relevant_indices[len(reorder_X_test)] = test_indices[index]
                        reorder_X_test.append(X_test[index])

                reorder_X_test = np.array(reorder_X_test)

                distances = [np.sqrt(np.sum((avgpoolarray - vector)**2)) for vector in reorder_X_test]

                for index in feedback.keys():
                    if len(nearest_indices)<10:
                                if len(nearest_indices)<10:
                                    nearest_indices.append(index)

                required_num = t - len(nearest_indices)
                print("Required Num: "+str(required_num))
                min_indices = np.argsort(distances)[:required_num]

                for index in min_indices:

                    nearest_indices.append(relevant_indices[index])

                print(str(len(nearest_indices))+" after score 1")

                if len(nearest_indices)<t:

                    relevant_indices = {}

                    reorder_X_test = []

                    for index in range(X_test.shape[0]):
                        if y_pred[index] == 0:
                            relevant_indices[len(reorder_X_test)] = test_indices[index]
                            reorder_X_test.append(X_test[index])

                    reorder_X_test = np.array(reorder_X_test)

                    distances = [np.sqrt(np.sum((avgpoolarray - vector)**2)) for vector in reorder_X_test]

                    for index in feedback.keys():
                        if feedback[index] == 0:
                            if len(nearest_indices)<t:
                                nearest_indices.append(index)

                    required_num = t - len(nearest_indices)
                    print("Required Num: "+str(required_num))
                    min_indices = np.argsort(distances)[:required_num]

                    for index in min_indices:

                        nearest_indices.append(relevant_indices[index])

                    print(str(len(nearest_indices))+" after score 0")

        #st.write(nearest_indices)

        st.markdown("Updated Results based on Feedback")

        show_ksimilar_list(nearest_indices,feature_collection,"")
    
    else:

        y_train = [y/3 for y in y_train]

        #st.write(y_train)

        st.markdown("Updated Results based on Feedback")

        y_test = []

        for row_index in tqdm(range(len(X_test))):

            sim_scores = []

            index = test_indices[row_index]

            train_indices = []

            for feedback_index in feedback.keys():

                sim_scores.append(similarity_collection.find_one({'_id':feedback_index})["avgpool_descriptor"][str(index)])
                train_indices.append(feedback_index)

            #print("Row Index: "+str(row_index)+" Sim Scores: "+str(sim_scores))

            for idx in range(len(y_train)):  
                sim_scores[idx] *= y_train[idx]

            #print("Prob Values: "+str(sim_scores))

            prob_value = np.max(sim_scores)

            y_test.append(prob_value)

        nearest_indices = []
        for feedback_index in feedback.keys():
            if feedback[feedback_index]/3 == 1:
                nearest_indices.append(feedback_index)

        req = t - len(nearest_indices)

        best_indices = np.argsort(y_test)[-req:][::-1]

        for indice in best_indices:
            nearest_indices.append(test_indices[indice])

        show_ksimilar_list(nearest_indices,feature_collection,"")

class DBScan:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit_predict(self, X):
        print("Fitting Dbscan Model!")
        self.labels = np.full(X.shape[0], -1)  # Initialize labels as unassigned (-1)
        cluster_label = 0

        for i in tqdm(range(X.shape[0])):
            if self.labels[i] != -1:
                continue  # Skip points already assigned to a cluster

            neighbors = self.find_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = 0  # Label as noise
            else:
                cluster_label += 1
                self.expand_cluster(X, i, neighbors, cluster_label)

        return self.labels

    def find_neighbors(self, X, center_idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[center_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, X, center_idx, neighbors, cluster_label):
        self.labels[center_idx] = cluster_label

        for neighbor in neighbors:
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_label
                new_neighbors = self.find_neighbors(X, neighbor)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)   


def calculate_distances(X):
    print("\n#IN CALCULATE DISTANCE")
    num_records = len(X)
    
    euclidean_distances = pairwise_distances(X)
    print("Shape of distance matrix: ", euclidean_distances.shape)
    print("Type of distance matrix: ", type(euclidean_distances))
    
    for i in range(num_records):
        if(i%100==0):
            print("Length euclidean_distances[",i,"]: ", len(euclidean_distances[i]))
            print("Average distance of sample ",i, " against other samples: ", sum(euclidean_distances[i])/num_records)
    average_distances = []
    
    cos_sim = cosine_similarity(X)
    
    for i in range(num_records):
        average_distances.append(sum(euclidean_distances[i])/num_records)
    
    return np.array(sum(average_distances)/num_records), euclidean_distances

def perform_mds(X):
    st.write("Started MDS")

    mod_path = Path(__file__).parent.parent
    mat_file_path = str(mod_path.joinpath("Classifiers","DBScan","mds.mat"))

    if((os.path.exists(mat_file_path))==False):
        print("Inside mds if")
        embedding = MDS(n_components=2, normalized_stress='auto')
        X_mds = embedding.fit_transform(X)
        st.write("Transformed X shape:", X_mds.shape)
        savedict = {
            'X_mds' : X_mds
        }
        scipy.io.savemat(mat_file_path, savedict)
    
    mds_data = scipy.io.loadmat(mat_file_path)
    X_mds = mds_data['X_mds']
    X_mds = np.array(X_mds)

    return X_mds

def preprocess_dbscan(images_resnet):
    scaler = MinMaxScaler()
    images = scaler.fit_transform(np.array(images_resnet))
    images = normalize(images, norm='l2')

    return images

def set_dbscan_params(n_clusters_):
    if(n_clusters_==5):
        epsilon = 0.8
        min_samples = 16

    else:
        epsilon = 0.7
        min_samples = 24

    return epsilon, min_samples

def get_dbscan_cluster_details(n_clusters_, n_noise_, labels):
    clusters = []

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    print("No. of instances in each label: ")

    for i in range(n_clusters_+1):
        clusters.append(2*np.where(labels==i)[0])
        print("Length of Cluster", i, ": ",len(clusters[i]))
        print("Cluster",i,":\n")
        n = list(labels).count(i)

    return clusters

def plot_dbscan_clusters(labels, unique_labels, images_MDS):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 8))

    for label, color in zip(unique_labels, colors):
        cluster_points = images_MDS[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], cmap=ListedColormap(colors), label=f'Cluster {label}', s=4)

    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plt.show())

def plot_image_clusters(clusters, feature_collection):
    for i in range(0,len(clusters)):
        size = 20
        if(len(clusters[i])<20):
            size = len(clusters[i])
        cluster = np.random.choice(clusters[i], size=size, replace=False)
        st.write("CLUSTER", i)
        show_ksimilar_list(cluster,feature_collection,"")

def dbscan_predict(feature_collection, odd_feature_collection, caltech101, images, true_labels, clusters, n_clusters_):
    even_images_resnet, even_true_labels, odd_images_resnet, odd_true_labels = get_data(feature_collection, odd_feature_collection)

    odd_images = preprocess_dbscan(odd_images_resnet)

    odd_predicted_labels = []

    build = []

    for i in range(n_clusters_+1):
        cluster = []
        for k in clusters[i]:
            cluster.append(images[int(k/2)])
        build.append(cluster)
        cluster = np.array(cluster)

    count = 0
    
    for idx in range(1, len(caltech101), 2):
        euclidean_distances = []
        odd_image = odd_images[int(idx/2)].reshape(1, -1)
    
        for i in range(len(clusters)):
            cluster = np.array(build[i])
            euclidean_distances.append(np.mean(pairwise_distances(odd_image, cluster)[0]))
        
        assigned_cluster_number = euclidean_distances.index(min(euclidean_distances)) 
    
        if((idx-1)%200==0):
            print("Assigned cluster for image: ", idx, "is ", assigned_cluster_number)
    
        most_similar_images = np.array(build[assigned_cluster_number])
    
        y = []
        for i in clusters[assigned_cluster_number]:
            y.append(true_labels[int(i/2)])
        
        y = np.array(y)
    
    
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(most_similar_images, y)
    
        prediction = neigh.predict(odd_image)
        odd_predicted_labels.append(prediction)

    
        if(odd_predicted_labels[int(idx/2)]==odd_true_labels[int(idx/2)]):
            count = count + 1
    
    accuracy = count/4388
    return np.round(accuracy*100, 2), odd_predicted_labels

def get_data(feature_collection, odd_feature_collection):
    mod_path = Path(__file__).parent.parent
    mat_file_path = mod_path.joinpath("LatentSemantics","")
    mat_file_path = str(f'{mat_file_path}{os.sep}')
    even_desc_path = mod_path.joinpath("LatentSemantics","arrays.mat")
    odd_desc_path = mod_path.joinpath("LatentSemantics","arrays_odd.mat")

    print("Mat file path:", mat_file_path, "Even mat file path: ", even_desc_path, "Odd mat file path: ", odd_desc_path)

    try:

        even_data = scipy.io.loadmat(str(even_desc_path))
        even_labels = even_data['labels']
        even_fc_features = np.array(even_data['fc_features'], dtype=np.uint8)

        odd_data = scipy.io.loadmat(str(odd_desc_path))
        odd_labels = odd_data['labels']
        odd_fc_features = np.array(odd_data['fc_features'], dtype=np.uint8)

        even_true_labels = []
        odd_true_labels = []

        for idx in range(len(even_labels)):
            even_true_labels.append(np.where(even_labels[idx]==1)[0][0])

        even_true_labels = [data[x] for x in np.array(even_true_labels).tolist()]
        
        
        for idx in range(len(odd_labels)):
            odd_true_labels.append(np.where(odd_labels[idx]==1)[0][0])
        
        odd_true_labels = [data[x] for x in np.array(odd_true_labels).tolist()]

    except (scipy.io.matlab.miobase.MatReadError, FileNotFoundError) as e:

        store_by_feature(str(mat_file_path),feature_collection)
        store_by_feature_odd(str(mat_file_path),odd_feature_collection)

        even_data = scipy.io.loadmat(str(even_desc_path))
        even_labels = even_data['labels']
        even_fc_features = even_data['fc_features']

        odd_data = scipy.io.loadmat(str(odd_desc_path))
        odd_labels = odd_data['labels']
        odd_fc_features = odd_data['fc_features']

    return even_fc_features, even_true_labels, odd_fc_features, odd_true_labels

def model_dbscan(caltech101, feature_collection, odd_feature_collection, data, n_clusters_ = 5):
    even_images_resnet, even_true_labels, odd_images_resnet, odd_true_labels = get_data(feature_collection, odd_feature_collection)

    print("Even images shape: ", even_images_resnet.shape)
    print("Odd images shape: ", odd_images_resnet.shape)
    print("Even data type: ", type(even_images_resnet))
    print("Odd data type: ", type(odd_images_resnet))
    print("Even labels shape: ", type(even_true_labels), len(even_true_labels))
    print("Odd labels shape: ", type(odd_true_labels), len(odd_true_labels))

    # print(even_true_labels[:500])
    # print(odd_true_labels[:500])

    # labels_arr = np.array(even_true_labels)
    # unique, counts = np.unique(labels_arr, return_counts=True)

    # labels_arr = np.asarray((unique, counts)).T
    # print(labels_arr)

    mod_path = Path(__file__).parent.parent
    pkl_file_path = str(mod_path.joinpath("Classifiers","DBScan","odd_predicted_labels"+str(n_clusters_)+".pkl"))

    if os.path.exists(pkl_file_path):

        accuracy, odd_predicted_labels, labels, unique_labels, images_MDS, clusters = load_pickle_file(pkl_file_path)

    else:

        images = preprocess_dbscan(even_images_resnet)
        print("Pre-Processing done")
        epsilon, min_samples = set_dbscan_params(n_clusters_)
        st.write("DBScan parameters:\nEpsilon = ", epsilon, "\nMin samples = ", min_samples)

        dbscan = DBScan(epsilon, min_samples)
        labels = dbscan.fit_predict(images)
        print("Finished fitting DBscan model!")
        unique_labels = np.unique(labels)
        n_noise_ = list(labels).count(0)

        clusters = get_dbscan_cluster_details(n_clusters_, n_noise_, labels)

        images_MDS = perform_mds(images)

        print(type(images_MDS))

        accuracy, odd_predicted_labels = dbscan_predict(feature_collection, odd_feature_collection, caltech101, images, even_true_labels, clusters, n_clusters_)

        pickle.dump((accuracy, odd_predicted_labels, labels, unique_labels, images_MDS, clusters), open(pkl_file_path, 'wb+'))
        
    st.write("Clusters Visualised on a 2-Dimensional MDS Space")
    plot_dbscan_clusters(labels, unique_labels, images_MDS)

    st.write("Images Visualised as Clusters")
    plot_image_clusters(clusters, feature_collection)

    otl = odd_true_labels
    opl = odd_predicted_labels

    confusion_matrix = np.zeros((101,101))
    odd_predicted_labels = np.array(odd_predicted_labels)

    for i in range(len(odd_predicted_labels)):
        for k, v in data.items():  
            if v == odd_predicted_labels[i]:
                odd_predicted_labels[i] = k

    for i in range(len(odd_true_labels)):
        for k, v in data.items():  
            if v == odd_true_labels[i]:
                odd_true_labels[i] = k

    for idx in range(len(odd_true_labels)):

        l = int(odd_true_labels[idx])
        c = int(odd_predicted_labels[idx])
        confusion_matrix[l][c]+=1

    with st.expander("Confusion Matrix for Classification"):
        st.write(confusion_matrix)

    display_scores(confusion_matrix,otl,opl,accuracy)

        
dataset_size = 8677
dataset_mean_values = [0.5021372281891864, 0.5287581550675707, 0.5458470856851454]
dataset_std_dev_values = [0.24773670511666424, 0.24607509728422117, 0.24912913964278197]
p = 512

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