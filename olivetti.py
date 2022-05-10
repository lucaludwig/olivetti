
# <h1> 2. Principal Component Analysis </h1>

# <h2>a) Consider Olivetti faces dataset and use a classical dimensionality reduction technique (e.g. PCA) while preserving 99% of the variance. Then compute the reconstruction error for each image.</h2>

# In[130]:

# Import libraries
import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
# Clustering libraries and methods used
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
# Libraries to create 3D plot using seaborn cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
# PCA libraries and methods used 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA


# In[43]:


olivetti = fetch_olivetti_faces()

olivetti.keys()
print('Sizes of the dataset:',olivetti.data.shape)
# Load into Pandas dataframe
pd.DataFrame(olivetti.data)


# In[44]:


#Defining the olivetti data and target features for PCA
x = olivetti.data
y = olivetti.target
#We choose 0.99 to keep 99% variance
pca =PCA(0.99)
x_pcatransformed = pca.fit_transform(x)
# Check the shape after the transformation
x_pcatransformed.shape


# In[30]:


#calculating the inverse error
x_reconstructed = pca.inverse_transform(x_pcatransformed)
print(x_reconstructed)
#Calculation of the inverse
error = np.square(x_reconstructed - x).mean(axis=1)

print(error.mean())


# <h2>b) Next, take some of the images you built using the dimensionality reduction technique and modi- fy/add some noise to some of the images using techniques such as rotate, flip, darken (you can use libraries such as scikit-image [2] etc. to do this) and look at their reconstruction error. You will notice that how much larger the reconstruction error is. </h2>

# In[31]:


#Function used to add noise
def apply_gaussian_noise(x, sigma=0.1):
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise
#Applying the noise function to the reconstructed images
X_added_noise = apply_gaussian_noise(x_reconstructed, 0.1)

#The new error after adding noise
newerror = np.square(x_reconstructed, x).mean()

print(newerror)


# <h2>c) Finally, plot all the 3 respective reconstructed images side by side (original image, image after PCA, image after PCA + noise) and compare the results.</h2>

# In[73]:


# Set the plot function
def plot_images(img1, img2, img3):    
    f, axarr = plt.subplots(1,3)
    plt.title('Original image (LEFT), Applying PCA (MIDDLE) After adding additional gaussian noise (RIGHT)')
    axarr[0].imshow(img1, cmap = "gray")
    axarr[1].imshow(img2, cmap = "gray")
    axarr[2].imshow(img3, cmap = "gray")
    
# Tested on multiple set of images
for i in range(3):
    # get the image
    original_img  = x[i, :]
    #print(original_img.shape)
    
    # reshape to display image 64 x 64 = 4096
    original_img_reshaped = original_img.reshape(64, 64)
    #print(original_img_reshaped.shape)
    
    second_pca_img = x_reconstructed[i, :]
    second_pca_img_reshaped = second_pca_img.reshape(64, 64)
    
    third_noise_img = X_added_noise[i, :]
    third_noise_img_reshaped = third_noise_img.reshape(64, 64)
    
    plot_images(original_img_reshaped, second_pca_img_reshaped, third_noise_img_reshaped)
