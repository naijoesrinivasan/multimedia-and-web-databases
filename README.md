# Multimedia and Web Databases Project

This project explores **image feature extraction**, **dimensionality reduction**, and **similarity search** techniques using the **Caltech101 dataset**. It is divided into two phases, focusing on feature descriptor extraction, clustering, classification, and relevance feedback. The project leverages **MongoDB**, the **Sol Supercomputer**, and **Streamlit** for deployment.

---

## Project Phases

### Phase 1: Image Feature Extraction and Similarity Search
- **Goal**:
   - Extract and store feature descriptors for images.
   - Implement similarity search to retrieve visually similar images.

- **Steps**:
   - **Feature Extraction**:
     - **Color Moments**: 900-dimensional descriptor from a 10x10 grid.
     - **Histograms of Oriented Gradients (HOG)**: 900-dimensional gradient histogram.
     - **ResNet Features**:
       - `ResNet-AvgPool-1024`: 1024-dimensional vector from the avgpool layer.
       - `ResNet-Layer3-1024`: 1024-dimensional vector from the third layer.
       - `ResNet-FC-1000`: 1000-dimensional vector from the fully connected layer.
   - **Storage**:
     - Extracted feature descriptors were stored in **MongoDB** for efficient querying.
   - **Similarity Search**:
     - Implemented distance measures to retrieve the top-k similar images based on feature descriptors.

- **Outcome**:
   - Created a robust system to extract image features and perform similarity searches using various feature models.

---

### Phase 2: Clustering, Classification, and Relevance Feedback
- **Goal**:
   - Perform advanced operations like clustering, classification, and relevance feedback using feature descriptors.
   - Optimize for high-dimensional data using dimensionality reduction and indexing.

- **Steps**:
   - **Clustering**:
     - Implemented **DBSCAN** from scratch to group images into clusters.
     - Visualized clusters as:
       - Colored point clouds in a 2D MDS (Multi-Dimensional Scaling) space.
       - Groups of image thumbnails.
   - **Classification**:
     - Built classifiers:
       - **m-Nearest Neighbors (m-NN)**.
       - **Decision Trees**.
       - **Personalized PageRank (PPR)**.
     - Evaluated classifiers using precision, recall, F1-score, and accuracy.
   - **Latent Semantic Analysis (LSA)**:
     - Extracted latent semantics for each label to reduce dimensionality and enhance similarity search.
   - **Locality Sensitive Hashing (LSH)**:
     - Implemented LSH for approximate nearest neighbors on high-dimensional data.
     - Optimized image searches for efficiency.
   - **Relevance Feedback**:
     - Developed relevance feedback systems:
       - **SVM-Based**: Reranks results based on user feedback.
       - **Probabilistic**: Refines search using probabilistic techniques.
     - Allowed users to tag images as "Very Relevant," "Relevant," or "Irrelevant" to improve future search results.
- **Outcome**:
   - Enhanced similarity search and classification using advanced clustering and indexing techniques.
   - Introduced interactive relevance feedback for improved user experience.

---

## Tech Stack
- **Programming Language**: Python
- **Frameworks/Libraries**:
  - Deep Learning: PyTorch, TorchVision
  - Numerical Computation: NumPy, SciPy
  - Database: MongoDB
  - Visualization: Streamlit
- **Computing Infrastructure**: Sol Supercomputer for expensive computations
- **Dataset**: Caltech101 (downloaded via TorchVision)

---

## Features
- **Feature Descriptors**:
  - Extracted various descriptors (Color Moments, HOG, ResNet features).
- **Similarity Search**:
  - Retrieved top-k similar images using distance measures.
- **Clustering**:
  - Grouped images into clusters using DBSCAN.
- **Classification**:
  - Implemented multiple classifiers and evaluated their performance.
- **Dimensionality Reduction**:
  - Applied LSA to optimize high-dimensional data.
- **Efficient Indexing**:
  - Built an LSH index for fast approximate nearest neighbor searches.
- **Relevance Feedback**:
  - Refined search results based on user feedback.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/naijoe-srinivasan/multimedia-and-web-databases.git
   cd multimedia-web-databases
   ```
2. Install required libraries
   ```bash
   pip install -r requirements.txt
   ```
3. Set up MongoDB:
    - Install and configure MongoDB locally or connect to a cloud instance.
    - Import the extracted feature descriptors into the MongoDB database.
4. Run the Streamlit app:
   ```bash
   streamlit run <task_number>.py
   ```
## Usage

- Feature Extraction:
  - Extract features from the Caltech101 dataset using the provided Python scripts.
- Clustering:
  - Visualize clusters in 2D MDS space or as grouped thumbnails.
- Similarity Search:
  - Input an image ID to find the top-k similar images.
- Classification:
  - Predict labels for test images using classifiers like m-NN or Decision Trees.
- Relevance Feedback:
  - Interact with the search system to refine results based on user feedback.


## Results
- Feature Descriptors:
  - Successfully extracted high-dimensional features for all images in the Caltech101 dataset.
- Similarity Search:
  - Achieved fast and accurate retrieval of visually similar images.
- Clustering:
  - Visualized meaningful image clusters using DBSCAN.
- Classification:
  - Achieved high precision and recall values for label prediction.
- Interactive Feedback:
  - Improved search relevance with SVM and probabilistic feedback systems.
## File Structure
- *code/*: Contains all the Python scripts for feature extraction, clustering, and classification.
- *outputs/*: Includes sample outputs (e.g., cluster visualizations, similarity searches).
- *report/*: Contains the project report summarizing methodologies and results.
- *app.py*: Streamlit application for user interaction.