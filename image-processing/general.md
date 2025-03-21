# ?Usages

* Visualization
* Image sharpening and restoration
* Image retrieval
* Measurement of patterns
* Image recognition

# SKimage 

It's a python library that enables easy ways to manipulate images

## Convert to grey scale

```
# Import the modules from skimage
from skimage import data, color

# Load the rocket image
rocket = data.rocket()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)

# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')
```

# Numpy to manipulate images

## flip 

```python 
# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Flip the image horizontally
seville_horizontal_flip = np.fliplr(seville_vertical_flip)
```

# Histograms

This allow us to know meaningful data about the image. You have to get one for each color channel if image is rgb. A

```python 
# Obtain the red channel
red_channel = image[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)

# Set title and show
plt.title('Red Histogram')
plt.show()
```
![Red histogram](./assets/color-hist.png)

# Thresholding

Partitioning and image into a foreground and a background.. Setting each pixel to 255(with) or 0 (white) based on a threshold. 

## Types 
* Global or histogram based: good for uniform backgrounds

```python
# Import the otsu threshold function
from skimage.filters import threshold_otsu

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

# Obtain the optimal threshold value with otsu
thresh =  threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the image
show_image(binary, 'Binary image')
```
![Global original](./assets/global_original.png)
![Global result](./assets/global_result.png)

* Local or adaptive: for uneven background illumination

You need to set a block size and an offset

```python 
# Import the local threshold function
from skimage.filters import threshold_local

# Set the block size to 35
block_size = 35

# Obtain the optimal local thresholding
local_thresh = threshold_local(page_image, block_size, offset=10)

# Obtain the binary image by applying local thresholding
binary_local = page_image > local_thresh

# Show the binary image
show_image(binary_local, 'Local thresholding')
```
![Local original](./assets/local_original.png)
![Local result](./assets/local_result.png)


## Try different methods

```python 
# Import the try all function
from skimage.filters import try_all_threshold

# Import the rgb to gray convertor function 
from skimage.color import rgb2gray

# Turn the fruits_image to grayscale
grayscale = rgb2gray(fruits_image)

# Use the try all method on the resulting grayscale image
fig, ax = try_all_threshold(grayscale, verbose=False)

# Show the resulting plots
plt.show()
```

![Different methods result](./assets/different_methods.png)
