# Fitlers 

Modify or enhance images. 

* Enhance
* Emphasize or remove features
* Smoothing
```python 
# Import Gaussian filter 
from skimage.filters import gaussian

# Apply filter
gaussian_image = gaussian(building_image, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")
```
![Gaussian original](./assets/gaussian_original.jpg)
![Gaussian result](./assets/gaussian_result.png)

* Sharpening
* Edge detection

```python 
# Import the color module
from skimage import color

# Import the filters module and sobel function
from skimage.filters import sobel

# Make the image grayscale
soaps_image_gray = color.rgb2grey(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")
```
![Sobel original](./assets/sobel_original.png)
![Sobel result](./assets/sobel_result.png)


# Contrast enhancement

You can use the histogram to detect contrast in an image. To calculate the contrast you can use the range of pixel values

![Low contrast histogram](./assets/low-contrast-histogram.png)

Small difference between pixel values. Depending in the

![High contrast histogram](./assets/high-contrast-histogram.png)

## Techniques

* Contrast stretching
* Histogram equalization (Standard, adaptive, limited adaptive)

![High contrast histogram](./assets/enhacement-techiniques.png)


```python 
# Import the required module
from skimage import exposure

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)

# Show the resulting image
show_image(xray_image_eq, 'Resulting image')
```

Adaptive uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than the rest of the image.

```python 
# Import the necessary modules
from skimage import data, exposure

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)

# Compare the original image to the equalized
show_image(original_image)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')
```

# Transformations

* rotation 
* rescale
* resize 

```python 
# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale

# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, multichannel=True)

# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")
```

The anti aliasing filter prevents the poor pixelation effect to happen, making it look better but also less sharp.

![](./assets/anti-aliasing-transform.png)
![](./assets/no-anti-aliasing-transform.png)

```python 
# Import the module and function
from skimage.transform import resize

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width),
anti_aliasing=True)

# Show the original and resized image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')
```

# Morphology

Dilatation: add pixels to the border of object
Erosion: Removes pixels to the border of object

## Structuring element 

We use a structuring elements (binary image) to process the image to determine the shape of the objects.

![](./assets/structuring-element-types.png)

The center is the origin of the structuring element

```python 
# Import the morphology module
from skimage import morphology

# Obtain the eroded shape 
eroded_image_shape = morphology.binary_erosion(upper_r_image) 

# Obtain the dilated image 
dilated_image = morphology.binary_dilation(world_image)

```