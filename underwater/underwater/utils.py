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
  content = requests.get(url)
  # unzip the content
  f = ZipFile(BytesIO(content.content))
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
  input:url to zipped file
  output: decoded series of 3D numpy arrays and the coded column(not used afterwards) 
  """
  df = read_csv_from_url(url)
  #ThumbnailImage = df.ThumbnailImage
  PreviewImage = df.PreviewImage
  to_array = PreviewImage.apply(lambda x: readb64(x))
  return to_array, PreviewImage

def series_range_0_1(series):
  """
  input: series with 0-255 range numpy arrays
  output: series with 0-1 range numpy arrays
  """
  import numpy as np
  series = series / 255
  return series

def reshape_image(image):
  """
  input: image as 3D numpy array of shape  (w, h, d)
  output:  image as 1D numpy array (w * h * d)
  """
  # Load Image and transform to a 2D numpy array.
  w, h, d = original_shape = tuple(image.shape)
  assert d == 3
  image_array = np.reshape(image, (w * h * d))
  return image_array

def reshape_series_images(series):
  """ 
  input: series of 3D images
  output: series of 1D images
  """
  series = series.apply(lambda x: reshape_image(x))
  return series

def visualizing(images):
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

