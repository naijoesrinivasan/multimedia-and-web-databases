import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path

def to_base64(image):
    from io import BytesIO
    import base64

    image_pil = Image.fromarray(image)
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_image_centered(image,idx):
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
            st.image(image=image, caption="ImageID: "+idx,channels="BGR", width = 300)
    
def display_feature_vector(vector, heading):

    with st.expander(heading):
        st.write(vector)
    
def display_color_moments(color_moments):
    
    with st.expander("Query Image Color Moments"):
        st.write(color_moments.tolist())
    
    
    
def display_hog(hog_descriptor, cell_size=(30, 10)):

    with st.expander("Query Image HOG Descriptor"):
        st.write(hog_descriptor)
    
    
def display_images(images, indices, similarity_scores, rows, cols, scorestring):
    
    k = len(images)
    
    if scorestring!="":

        with st.container():
            for i in range(0,k,5):
                cols = st.columns(5)

                with cols[0]:
                    if i<k:
                        st.image(images[i],caption = "Rank: "+str(i+1)+"\nImage ID: "+str(indices[i])+"\n"+scorestring+str(similarity_scores[i]))

                with cols[1]:
                    if i+1<k:
                        st.image(images[i+1],caption = "Rank: "+str(i+2)+"\nImage ID: "+str(indices[i+1])+"\n"+scorestring+str(similarity_scores[i+1]))

                with cols[2]:
                    if i+2<k:
                        st.image(images[i+2],caption = "Rank: "+str(i+3)+"\nImage ID: "+str(indices[i+2])+"\n"+scorestring+str(similarity_scores[i+2]))

                with cols[3]:
                    if i+3<k:
                        st.image(images[i+3],caption = "Rank: "+str(i+4)+"\nImage ID: "+str(indices[i+3])+"\n"+scorestring+str(similarity_scores[i+3]))

                with cols[4]:
                    if i+4<k:
                        st.image(images[i+4],caption = "Rank: "+str(i+5)+"\nImage ID: "+str(indices[i+4])+"\n"+scorestring+str(similarity_scores[i+4]))

    else:

        with st.container():
            for i in range(0,k,5):
                cols = st.columns(5)

                with cols[0]:
                    if i<k:
                        st.image(images[i],caption = "Rank: "+str(i+1)+"\nImage ID: "+str(indices[i]))

                with cols[1]:
                    if i+1<k:
                        st.image(images[i+1],caption = "Rank: "+str(i+2)+"\nImage ID: "+str(indices[i+1]))

                with cols[2]:
                    if i+2<k:
                        st.image(images[i+2],caption = "Rank: "+str(i+3)+"\nImage ID: "+str(indices[i+2]))

                with cols[3]:
                    if i+3<k:
                        st.image(images[i+3],caption = "Rank: "+str(i+4)+"\nImage ID: "+str(indices[i+3]))

                with cols[4]:
                    if i+4<k:
                        st.image(images[i+4],caption = "Rank: "+str(i+5)+"\nImage ID: "+str(indices[i+4]))



def show_ksimilar(k_similar,collection,scorestring):
    images = []
    indices = []
    similarity_scores = []
    count = len(k_similar.keys())
    rows = int(count/5)
    if (rows*5)<count: rows+=1
    cols = 5
    for index in k_similar.keys():
        document = collection.find_one({'_id': int(index)})
        images.append(cv2.cvtColor(np.array(document['image'], dtype=np.uint8), cv2.COLOR_BGR2RGB))
        indices.append(int(index))
        similarity_scores.append(k_similar[str(index)])
        
    display_images(images, indices, similarity_scores, rows, cols,scorestring)    

def show_ksimilar_list(k_similar,collection,scorestring):
    images = []
    indices = []
    similarity_scores = []
    count = len(k_similar)
    rows = int(count/5)
    if (rows*5)<count: rows+=1
    cols = 5
    for index in k_similar:
        document = collection.find_one({'_id': int(index)})
        images.append(cv2.cvtColor(np.array(document['image'], dtype=np.uint8), cv2.COLOR_BGR2RGB))
        indices.append(int(index))
        
    display_images(images, indices, similarity_scores, rows, cols,scorestring)    