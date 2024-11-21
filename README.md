# K-Means Clustering: Custom Implementation for Data Clustering and Visualization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-purple.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-yellow.svg)

## 📈 Overview

This **custom implementation of the K-Means clustering algorithm** provides a fundamental yet flexible approach for partitioning data into clusters. Designed for simplicity and customization, it supports applications such as data preprocessing, exploratory analysis, and unsupervised learning tasks. By iteratively updating centroids and assignments, this implementation converges to meaningful data groupings while offering visual insights through plots.

## 🚀 Key Features

### **Initialization**
- Supports **random initialization** of centroids within the data range.
- Offers options to specify the number of clusters (`k`) and iteration limit.

### **Clustering**
- Iteratively refines cluster assignments and updates centroids.
- Computes within-cluster variance to monitor convergence.
- Implements early stopping based on minimal centroid movement.

### **Visualization**
- Includes tools for plotting clustered data points, centroids, and decision boundaries for 2D datasets.
- Allows visualization of clustering progress through iterative snapshots.

## 🎨 Visual Examples

### **1. Clustering of Synthetic Dataset**  
This scatterplot shows how K-Means clusters synthetic 2D data points, assigning each cluster a unique color and marking centroids with larger symbols.

<img src="https://github.com/rubythedev/k-means/blob/main/images/kmeans_cluster.png" width="400">

### **2. Convergence Plot**  
This plot visualizes the decrease in within-cluster variance over iterations, illustrating the algorithm's convergence.

<img src="https://github.com/rubythedev/k-means/blob/main/images/convergence_plot.png" width="400">

### **3. K-Means Clustering on Baby Bird Image**  
This image shows the original and compressed version of a baby bird image after applying K-Means clustering.

<img src="https://github.com/rubythedev/k-means/blob/main/images/baby_bird.png" width="400">

### **4. K-Means Clustering on Angelina Jolie Image**  
This image shows the original and compressed version of an image of Angelina Jolie after applying K-Means clustering.

<img src="https://github.com/rubythedev/k-means/blob/main/images/angelina.png" width="400">

## 🛠️ Technologies & Skills

- **Programming Languages:** 
  - [Python 3.x](https://www.python.org/) for numerical computations and visualization.

- **Libraries & Frameworks:** 
  - [NumPy](https://numpy.org/) for matrix operations and data manipulation.
  - [Matplotlib](https://matplotlib.org/) for 2D plotting and visualization.
  - [SciPy](https://scipy.org/) for distance computations and advanced mathematical functions.
  - [Pandas](https://pandas.pydata.org/) for data handling, preprocessing, and exploratory data analysis.
  - [Scikit-learn](https://scikit-learn.org/) for synthetic data generation (e.g., using `make_blobs`).
  - [Seaborn](https://seaborn.pydata.org/) for enhanced visualizations and aesthetic plots (if applicable).

- **Machine Learning Techniques:** 
  - **Unsupervised Learning:** Clustering without labels using centroid-based methods.
  - **Euclidean Distance:** Computing data-point-to-centroid distances for assignments.

- **Data Visualization:** 
  - Scatterplots to show cluster assignments.
  - Line plots for convergence monitoring.

## 💡 Why K-Means?

K-Means is a widely used clustering algorithm due to its simplicity and efficiency. This implementation showcases its ability to group data points into clusters, providing insights into structure and patterns within datasets. Its modular design enables experimentation with datasets of varying complexity.

## 🚀 Getting Started

This project demonstrates the implementation of the **K-Means Clustering** algorithm. You can use it for both numeric datasets (like CSV files) and images. Follow the steps below to get started.

### Prerequisites

Before running the project, make sure you have the following libraries installed in your Python environment:  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **SciPy**  

Install them using `pip`:

```python
pip install numpy pandas matplotlib scipy
```

### **1. Import Required Libraries**

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import kmeans
import pandas as pd
from matplotlib.image import imread
```

### 2. Load Your Data

#### For Numerical Data:
To start, load your dataset (e.g., a CSV file) using Pandas. Then, convert it into a NumPy array to prepare it for the K-Means algorithm.

```python
# Load a CSV file as a DataFrame
df = pd.read_csv('data/your_dataset.csv')

# Convert the DataFrame to a NumPy array
your_data = df.values

# Preview the data
print(your_data)
```

#### For Image Data:
To start, load your dataset (e.g., a CSV file) using Pandas. Then, convert it into a NumPy array to prepare it for the K-Means algorithm.

```python
# Load an image
image = imread('data/your_image.jpg')

# Flatten the image
def flatten(img):
    '''Flattens an image to N 1D vectors.'''
    num_rows, num_cols, rgb = img.shape    
    flattened_img = img.reshape(num_rows * num_cols, rgb)
    return flattened_img

flattened_image = flatten(image)

# Preview the flattened image shape
print(flattened_image.shape)
```

### 3. Prepare Your Data (For Image or Numerical Data)

#### For Image Data:
If you're working with images, you need to load the image and flatten it into 1D vectors, which K-Means can use as data points for clustering. This is necessary because each pixel in the image is represented by its RGB (Red, Green, Blue) values, and these need to be treated as individual data points in the clustering process.

```python
# Load an image
image = imread('data/your_image.jpg')

# Flatten the image to 1D vectors (each pixel's RGB values)
def flatten(img):
    '''Flattens an image to N 1D vectors.'''
    num_rows, num_cols, rgb = img.shape    
    return img.reshape(num_rows * num_cols, rgb)

# Flatten the image
flattened_image = flatten(image)

# Preview the flattened image shape (total pixels x 3 RGB values)
print(flattened_image.shape)
```

### 4. Initialize K-Means Class

Now that you have your data prepared, it's time to initialize the **K-Means** class, where you'll specify the number of clusters (k) and apply the algorithm to your data.

#### For Numerical Data:
```
# Create an instance of the KMeans class with the input data
cluster = kmeans.KMeans(your_data)

# Specify the number of clusters
k = 3

# Initialize the centroids
init_centroids = cluster.initialize(k)

# Preview the initial centroids
print(init_centroids)
```

#### For Image Data:
```python
# Create an instance of the KMeans class with the flattened image data
image_cluster = kmeans.KMeans(flattened_image)

# Specify the number of clusters
k = 5

# Initialize the centroids
image_init_centroids = image_cluster.initialize(k)

# Preview the initial centroids
print(image_init_centroids)
```

### 5. Assign Data Points to Clusters (Update Labels)

After initializing the centroids, the next step is to assign each data point to its closest centroid, thereby forming clusters.

#### For Numerical Data:
```python
# Assign data points to the nearest centroids, producing cluster labels
new_labels = cluster.update_labels(init_centroids)

# Preview the new labels (which data points belong to which clusters)
print(new_labels)
```

#### For Image Data:
```python
# Assign image pixels (data points) to the nearest centroids
image_new_labels = image_cluster.update_labels(image_init_centroids)

# Preview the new labels for image data
print(image_new_labels)
```

### 6. Update Centroids

The next step is to update the centroids based on the mean of the data points assigned to each cluster. 

#### For Numerical Data:
```python
# Update the centroids and calculate the difference between the new and previous centroids
new_centroids, diff_from_prev_centroids = cluster.update_centroids(k, new_labels, init_centroids)

# Preview the new centroids
print(new_centroids)
```

#### For Image Data:
```python
# Update the centroids and calculate the difference for image data
image_new_centroids, image_diff_from_prev_centroids = image_cluster.update_centroids(k, image_new_labels, image_init_centroids)

# Preview the updated centroids for the image data
print(image_new_centroids)
```

### 7. Perform the Clustering Process

Now, you're ready to run the K-Means clustering process for your data with the chosen number of clusters.

#### For Numerical Data:
```python
# Perform the clustering process
cluster.cluster(k)

# Preview the clustered data
cluster.plot_clusters()

# Display the plot
plt.show()
```

#### For Image Data:
```python
# Perform the clustering process for image data
image_cluster.cluster(k)

# Replace colors in the image with the centroid colors
image_cluster.replace_color_with_centroid()

# Reshape the compressed image back to its original shape
compressed_image = np.reshape(image_cluster.data, image.shape)

# Plot the original and compressed images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot compressed image
ax[1].imshow(compressed_image)
ax[1].axis('off')
ax[1].set_title('Compressed Image')

plt.show()
```

### 8. Evaluate Your Model: The Elbow Method

The **Elbow Method** is a useful technique to determine the optimal number of clusters (`k`) for your data. This method plots the sum of squared distances (inertia) against various values of `k` to find the point where adding more clusters provides diminishing returns.

#### For Numerical Data:
```python
# Set the maximum number of clusters to evaluate
max_k = 10

# Generate the elbow plot
cluster.elbow_plot(max_k)
plt.title("Elbow Plot (Numerical Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
```

#### For Image Data:
```python
# Set the maximum number of clusters to evaluate
max_k = 10

# Generate the elbow plot for the image
image_cluster.elbow_plot(max_k)
plt.title("Elbow Plot (Image Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
```

### 9. Batch Clustering (Optional)

If you want to experiment with batch clustering, you can perform clustering in batches rather than iterating over the entire dataset all at once. This can be useful for large datasets.

#### For Numerical Data:
```python
# Perform batch clustering
cluster.cluster_batch(k=3, n_iter=10)

# Plot the clustered data
cluster.plot_clusters()

# Display the plot
plt.show()
```

#### For Image Data:
```python
# Perform batch clustering for image data
image_cluster.cluster_batch(k=5, n_iter=20)

# Replace colors in the image with the centroid colors
image_cluster.replace_color_with_centroid()

# Reshape the compressed image back to its original shape
compressed_image_batch = np.reshape(image_cluster.data, image.shape)

# Plot the original and compressed images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot compressed image
ax[1].imshow(compressed_image_batch)
ax[1].axis('off')
ax[1].set_title('Compressed Image (Batch Clustering)')

plt.show()
```

### 10. Final Thoughts

Once the K-Means clustering process is complete, you can use the results to analyze the data. The visualizations will give you insights into how your data is grouped and whether it corresponds to any meaningful patterns. Keep in mind that K-Means is sensitive to initial centroid placement and may require multiple runs for stable results.


## 📈 Example Project: K-Means Clustering on Images

This section demonstrates the application of K-Means clustering to an image (Angelina Jolie image) to compress its colors by reducing the number of unique RGB values. The following steps show how K-Means can be used for image compression.

### **1. Load the Image**

First, load the image using the `imread` function from `matplotlib` and display the original image.

```python
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Load the image
angelina_jolie_image = imread('data/angelina_jolie.jpg')

# Display the original image
plt.imshow(angelina_jolie_image)
plt.axis("off")
plt.show()
```

### 2. Flatten the Image

K-Means requires the data in a 1D format, so flatten the 3D image (height, width, RGB) into a 2D array (each row represents a pixel, and columns represent the RGB values).

```python
import numpy as np

# Flatten the image
def flatten(img):
    '''Flattens an image to N 1D vectors.'''
    num_rows, num_cols, rgb = img.shape    
    return img.reshape(num_rows * num_cols, rgb)

# Flatten the image
angelina_flattened_image = flatten(angelina_jolie_image)
```

### 3. Apply K-Means Clustering
Now, initialize K-Means clustering and apply it to compress the image by reducing the number of unique colors (RGB values).

```python
import kmeans

# Set a random seed for reproducibility
np.random.seed(0)

# Set the number of clusters (k)
k = 5

# Create a KMeans instance and apply clustering
angelina_image = kmeans.KMeans(angelina_flattened_image)
angelina_image_init_centroids = angelina_image.initialize(k)
angelina_image_new_labels = angelina_image.update_labels(angelina_image_init_centroids)
angelina_image_new_centroids, angelina_image_diff_from_prev_centroids = angelina_image.update_centroids(k, angelina_image_new_labels, angelina_image_init_centroids)
angelina_image.cluster(k=k, max_iter=20)
angelina_image.replace_color_with_centroid()
angelina_compressed_image_reshaped = np.reshape(angelina_image.data, angelina_jolie_image.shape)
```

### 4. Display the Original and Compressed Image Side by Side
Finally, we plot the original and the compressed images side by side for comparison.

```python
# Plotting both images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot original image
ax[0].imshow(angelina_jolie_image)
ax[0].axis('off')
ax[0].set_title('Original Image')

# Plot compressed image
ax[1].imshow(angelina_compressed_image_reshaped)
ax[1].axis('off')
ax[1].set_title('Compressed Image')

plt.show()
```

### 5. Result
By using K-Means clustering with k=5, the image's color palette is reduced to only 5 unique colors, resulting in a compressed version of the original image. This can help in reducing the size of the image for storage or further analysis.

In this example, K-Means has compressed the image by clustering similar colors into 5 centroids.
