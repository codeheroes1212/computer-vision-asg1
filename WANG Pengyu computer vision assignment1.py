import cv2
import pytesseract
img = cv2.imread("D:\computer vision\image3.png")
# Define custom configuration options for Tesseract OCR
custom_config = r'--oem 3 --psm 6'
# Use Tesseract to extract text from the image with the specified configuration
pytesseract.image_to_string(img, config=custom_config)

cv2.imshow('result', img)
cv2.waitKey()

import cv2
import numpy as np
img = cv2.imread('D:\computer vision\image3.png')

# Function to convert the image to grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Function to remove noise from the image using median blurring
def remove_noise(image):
    return cv2.medianBlur(image,5)
# Function to apply thresholding to the image
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Function to dilate the image
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
# Function to erode the image
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)
# Function to perform opening (erosion followed by dilation) on the image
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# Function to detect edges in the image using Canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)
# Function to correct the skew of the image
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Function to match a template image within another image
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Apply various image processing functions to the original image
image = cv2.imread('D:\computer vision\image3.png')
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

img = gray
# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)
cv2.imshow('result',img)
cv2.waitKey()

img = thresh
# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)
cv2.imshow('result',img)
cv2.waitKey()

img = opening
# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)
cv2.imshow('result',img)
cv2.waitKey()

img = canny
# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)
cv2.imshow('result',img)
cv2.waitKey()

import cv2
import pytesseract
from pytesseract import Output
img = cv2.imread('D:\computer vision\image3.png')
# Extract text data from the image using pytesseract
d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

import cv2
import pytesseract
img = cv2.imread('D:\computer vision\image3.png')
# Get the height, width, and channels of the image
h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
cv2.imshow('result',img)
cv2.waitKey()

# Extract bounding boxes for each detected text element
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('result',img)
cv2.waitKey()


import re
import cv2
import pytesseract
from pytesseract import Output
img = cv2.imread('D:\computer vision\image3.png')
d = pytesseract.image_to_data(img, output_type=Output.DICT)
keys = list(d.keys())
target_text ='submission'
not_graded_pattern = re.compile(re.escape(target_text), re.IGNORECASE)
# Iterate through all detected text elements
for i in range(len(d['text'])):
    # Check if the confidence level is above 60
    if int(d['conf'][i]) > 60:
         # If the current text box contains the target text
         if re.search(not_graded_pattern, d['text'][i]):
            # Get the coordinates and dimensions of the bounding box
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            # Draw a green rectangle around each detected text element on the image
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#cv2.imshow('result', img)
#cv2.waitKey()

custom_config = r'--oem 3 --psm 6 outputbase digits'
print(pytesseract.image_to_string(img, config=custom_config))

def read_text_from_image(image):
  """Reads text from an image file and outputs found text to text file"""
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Perform OTSU Threshold
  ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  # Create a rectangular kernel for dilation
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
  # Dilate the thresholded image
  dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
  # Find contours in the dilated image
  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  # Make a copy of the original image
  image_copy = image.copy()


  # Iterate through each contour
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image_copy[y : y + h, x : x + w]
    file = open("results.txt", "a")
    text = pytesseract.image_to_string(cropped)
    file.write(text)
    file.write("\n")
  file.close()
image = cv2.imread("D:\computer vision\image3.png")
read_text_from_image(image)

# OCR results
cv2.imshow('result',img)
f = open("results.txt", "r")
lines = f.readlines()
lines.reverse()
for line in lines:
    print(line)
f.close()