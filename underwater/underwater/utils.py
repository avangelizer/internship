"""
This file contains functions that are used in the notebooks
It serves to declatter the notebooks.

"""

def read_csv_from_url(url):
  """
  input: url to a zip file
  role: unzips a file given the url without downloading it and returns the dataframe
  returns: a dataframe
  
  """
  import requests
  import pandas as pd
  from io import BytesIO
  from zipfile import ZipFile
  from gzip import GzipFile
  content = requests.get(url)
  
  try:
    # unzip the content
    f = ZipFile(BytesIO(content.content))
  except: 
    # unzip the content
    f = GzipFile(fileobj=BytesIO(content.content))
  else:
    print("not .zip and .gz")
  print(f.namelist())
  with f.open(f.namelist()[0], 'r') as g: 
    df = pd.read_csv(g)
  return df
def read_csv_from_url_gz(url):
  """
  input: url to a csv file zipped(.gz)
  role: unzips a file given the url without downloading it and returns the dataframe
  returns: a dataframe
  """
  import requests
  import pandas as pd
  from io import BytesIO
  content = requests.get(url)
  # unzip the content
  f = gzip.GzipFile(fileobj=BytesIO(content.content))
  print(f.namelist())
  with f.open(f.namelist()[0], 'r') as g: 
    df = pd.read_csv(g)
  return df
def decode_base64(column):
  """
  input: series where each element is an image coded in base64 with the format: "base64:XXXXXXX"
  role: decodes the relevant part (after:) 
  output: returns a series with io.BytesIO objects
  """
  import base64
  from io import BytesIO
  column_decoded = column.apply(lambda x: BytesIO(base64.b64decode(x.split(":")[1])))
  return column_decoded


def readb64(image_base64):
  """
  input: image coded in base64 with the format: "base64:XXXXXXX"
  role: decodes the relevant part (after:) 
  output: returns an RGB image
  """
  import base64
  import cv2
  import numpy as np
  encoded_data = image_base64.split(':')[1]
  nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8) 
  imageBGR = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  imageRGB = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
  return imageRGB

def reshaping_2D(image):
  """
  input: 3D numpy array
  output: 2D numpy array
  """
  x, y, z = image.shape
  image_2d = image.reshape(x*y, z)
  return image_2d

def load_series(url):
  """
  input:url to zipped file (.zip)
  output: decoded series of 3D numpy arrays and the coded column(not used afterwards) 
  """
  df = read_csv_from_url(url)
  print(df.columns)
  PreviewImage = df.PreviewImage
  to_array = PreviewImage.apply(lambda x: readb64(x))
  return to_array


def series_range_0_1(series):
  """
  input: series with 0-255 range numpy arrays
  output: series with 0-1 range numpy arrays
  """
  series = series / 255
  return series

def reshape_image_3D_2D(image):
  """
  input: image as 3D numpy array of shape  (w, h, d)
  output:  image as 1D numpy array (w * h * d)
  """
  # Load Image and transform to a 2D numpy array.
  w, h, d = original_shape = tuple(image.shape)
  assert d == 3
  image_array = np.reshape(image, (w * h * d))
  return image_array

def reshape_series_images_3D_2D(series):
  """ 
  input: series of 3D images
  output: series of 1D images
  """
  series = series.apply(lambda x: reshape_image_3D_2D(x))
  return series
def reshape_image(image):
  # Load Image and transform from 2D to a 1D numpy array.
  #w, h, d = original_shape = tuple(image.shape)
  w, h= original_shape = tuple(image.shape)
  #assert d == 3
  #image_array = np.reshape(image, (w * h * d))
  image_array = np.reshape(image, (w * h))
  return image_array
def reshape_series_images(series):
  """ 
  input: series of 2D images
  output: series of 1D images
  """
  series = series.apply(lambda x: reshape_image(x))
  return series

def visualizing_3D(images):
  """
  input: series (shape: (n_samples,n_features) of 1D images 
  output: visualize images after reshaping them
  """
  #images is 3D array
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(images.shape[0], 1, figsize=(20, 20))

  for axi, image in zip(ax.flat, images):
      image =image.reshape(720, 960, 3 )
      axi.set(xticks=[], yticks=[])
      axi.imshow(image, interpolation='nearest', cmap=plt.cm.binary)

  pass


def visualizing(images):
  #images is 2D array
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(images.shape[0], 1, figsize=(20, 20))

  for axi, image in zip(ax.flat, images):
      image =image.reshape(256, 256)
      axi.set(xticks=[], yticks=[])
      axi.imshow(image, interpolation='nearest', cmap=plt.cm.binary)

  
  pass

def resizing_img(img, shape=(256, 256)):
  """
  input:  a numpy array
  output: resized array in the desired shape
  """
  import cv2
  import numpy as np
  res = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA) 
  return res

def resizing_imgs_series(series, shape=(256, 256)):
  """
  series where each row is a numpy array (image) 
  """
  res = series.apply(lambda x: resizing_img(x,shape)) 
  return res
def image_to_gray(image):
  """
  input: a numpy array(image in RGB)
  output:a numpy array(image in grayscale)
  """
  import cv2
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def to_gray(series):
  """
  input: series where each row is a numpy array(image in RGB)
  output:series where each row is a numpy array(image in grayscale)
  """
  return series.apply(lambda x: image_to_gray(x))

def visualize_kmeans(y, x,cluster_centers_):
  """
  input:
  -y: the result of kmeans.fit_predict()
  -x: the input data of shape(n_samples, n_features)
  -cluster_centers_: the result of kmeans.fit().cluster_centers_
  """
  import matplotlib.pyplot as plt
  #Visualising the clusters
  plt.scatter([x[y_kmeans == 0, 0]], [x[y_kmeans == 0, 1]], s = 100, c = 'red', label = '0')
  plt.scatter([x[y_kmeans == 1, 0]], [x[y_kmeans == 1, 1]], s = 100, c = 'blue', label = '1')
  
  #Plotting the centroids of the clusters
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

  plt.legend()
  pass

def elbow_plot():
  #For future use
  #Taken from: https://www.kaggle.com/tonzowonzo/simple-k-means-clustering-on-the-iris-dataset
  #Finding the optimum number of clusters for k-means classification
  from sklearn.cluster import KMeans
  wcss = []

  for i in range(1, 11):
      kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
      kmeans.fit(x)
      wcss.append(kmeans.inertia_)

  #Plotting the results onto a line graph, allowing us to observe 'The elbow'
  plt.plot(range(1, 11), wcss)
  plt.title('The elbow method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS') #within cluster sum of squares
  plt.show()
  pass
