# Import the necessary libraries
from PIL import Image
from numpy import asarray
import numpy


# load the image and convert into
# numpy array
img = Image.open('threshold.png')
numpydata = asarray(img)
np_img = numpy.array(img)
  
print(np_img.shape)

# data
print(numpydata)
