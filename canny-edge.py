import csv 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
from skimage import io
from scipy import ndimage
from PIL import Image as im 

""" 
The purpose of this file is to convert all of the data from greyscale images, 
to binary images where 255 represents an edge detected by the canny edge detector.

We obtained most of source code for this algorithm from https://www.programmersought.com/article/1750536116/
"Canny edge detection method in python - computer vision"
"""

train_read_path = './archive/sign_mnist_train.csv'
test_read_path = './archive/sign_mnist_test.csv'
train_write_path = './archive/canny_train.csv'
test_write_path = './archive/canny_test.csv'

# Gaussian 5x5 kernal 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

# Gradient
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)


# none max suppresison
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z


# threshold
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)


# hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
    
    
# TRAINING DATA
TITLE_ROW = None
canny_rows = []
with open(test_read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print("Working on image")
        print(line_count)
        if line_count == 0:
            TITLE_ROW = row
        else:
            label = row[0]
            image = np.asarray(row[1:]).reshape((28, 28))
            image.astype(np.float)
            image = (image).astype(np.uint8)
            data = im.fromarray(image, 'L')
            # CANNY EDGE DETECTOR
            #after guassian filter
            blurred = ndimage.gaussian_filter(data, sigma=1)
            #gradient
            sobel, theta = sobel_filters(blurred)
            #non max suppression
            non_max_sup = non_max_suppression(sobel, theta)
            #threshold
            res, weak, strong = threshold(non_max_sup)
            #htsteresis
            canny_img = hysteresis(res,weak)
            canny_row = canny_img.flatten().tolist()
            canny_row.insert(0, label)
            canny_rows.append(canny_row)
        line_count += 1
print(len(canny_rows))

# Write the information
with open(test_write_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(TITLE_ROW)
    for row in canny_rows:
        print("Writing line")
        writer.writerow(row)
        
# TESTING DATA
# Read the converted images by row to 
with open('innovators.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SN", "Name", "Contribution"])
    writer.writerow([1, "Linus Torvalds", "Linux Kernel"])
    writer.writerow([2, "Tim Berners-Lee", "World Wide Web"])
    writer.writerow([3, "Guido van Rossum", "Python Programming"])
    
if __name__ == '__main__':

    img = io.imread('Letter_A.png', as_gray = True)
    #after guassian filter
    blurred = ndimage.gaussian_filter(img, sigma=1)

    #gradient
    sobel, theta = sobel_filters(blurred)

    #non max suppression
    non_max_sup = non_max_suppression(sobel,theta)

    #threshold
    res , weak, strong = threshold(non_max_sup)


    #htsteresis
    img1 = hysteresis(res,weak)

    #show image
    plt.imshow(img1,cmap="gray")

    plt.show()
