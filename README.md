# Computer-vision-asg1：
This is a project about using a variety of image processing tools for image processing and text extraction of uploaded picture.
# Background-introduction：
In the context of rapid development of information technology, text detection technology, as an important component of computer vision and artificial intelligence fields, is widely used in document analysis, license plate recognition, scene text detection, and many other fields. This article will analyze in detail the performance of a text detection system based on OpenCV and Tesseract provided by the code, and give corresponding evaluation results.
# This-code-consists-of-the-following-operations:
#	Setup Environment
1. Import required libraries, including OpenCV (cv2) and Tesseract OCR (pytesseract).
2. Read the image file.
# Pre-process the Image
3. A series of image processing functions are defined, including gray-scale, median filter denoising, thresholding, expansion, corrosion, open operation, Canny edge detection and tilt correction.
4. Invoke Tesseract OCR with custom configuration options to extract text from the image.
5. Apply various image processing functions to the original image and display the results.
# Text Detection
6. Use Tesseract OCR advanced functions, such as extracting bounding box information, matching templates, etc.
7. Use regular expressions to filter specific text (such as "submission") and highlight it on the image.
# Extract and Display Text
8. Output the extracted text to the console or text file.

# Project-summary
The main function of this code is to preprocess the given image to improve the accuracy of text recognition, and extract the text information in the image through Tesseract OCR. At the same time, it also provides some additional functions such as edge detection, template matching, etc., to help further analyze and process images.
