# SPEAR - Storm Parameter Extraction and Analysis Resource

import numpy as np
from matplotlib.patches import Ellipse
import cv2
from scipy import ndimage
import h5py # needs conda/pip install h5py
import math


DATA_PATH    = 'D:\SEVIR Data\data'
CATALOG_PATH = 'D:\SEVIR Data/CATALOG.csv'

# Image inputs are 2D arrays of values 0-255 representing intensity of precipitation.
# Binarization is done in 2 ways: 
    # Weighted - Any pixel with value larger than threshold adds number of points equal to pixel value to new binarized XY vectors with value of 1.
        # Accounts for pixel intensities with frequency of pointsk
    # Unweighted - Any pixel with value larger than threshold is added to new binarized XY vectors with value of 1.



# Input: Image in 2D array format, binarization threshold (0-255)
# Output: Weighted{Semi-Major Axis length (pixels), Semi-Minor Axis length (pixels), Angle of Semi-Major Axis above horizontal}, Unweighted{''}
# NOTE: Revised version calculates ellipse for 2 standard deviations, capturing ~95% of sotrm data in image
def findStormParams(originalImage, threshold, nSTD):
    image = convertMissingToZero(originalImage)
    imageBinaryY, imageBinaryX = np.where(image>threshold)
    if imageBinaryX.size == 0:
        return -1

    # Code from https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipse-in-a-scatterplot-using-matplotlib
    
    covariance = np.cov(imageBinaryX, imageBinaryY)
    # eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # eigenvalues = np.sqrt(eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:,order]
    angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
    semiMajor, semiMinor = nSTD * np.sqrt(eigenvalues)

    
    # For-loop determines standard deviations used for ellipse (2 currently)
    # semiMajor = (eigenvalues[0]*j * nSTD)
    # semiMinor = (eigenvalues[1]*j * nSTD)
    # angle = (np.rad2deg(np.arccos(eigenvectors[0, 0])))

    com = findCenter(image, threshold)
    # comX = com[0]
    # comY = com[1]
    # velocity = findCOMVelocity(com)
    # angleVelocity = findAngularVelocity(weightedAngle)


    non_zero_indices_x, non_zero_indices_y = np.nonzero(originalImage)
    non_zero_values = originalImage[non_zero_indices_x, non_zero_indices_y]
    non_zero_values = non_zero_values[(non_zero_values > threshold)]
    average = np.average(non_zero_values)
    
    # intensitySum = imageBinaryX.size
    ellipseArea = math.pi * semiMajor * semiMinor
    sum = np.sum(originalImage)
    if semiMajor > 0 and semiMinor > 0:
        ellipseIntensity = sum / ellipseArea
        axisRatio = semiMajor / semiMinor
    else:
        ellipseIntensity = -1
        axisRatio = -1
    if semiMinor > semiMajor:
        temp = semiMinor
        semiMinor = semiMajor
        semiMajor = temp
    
    return semiMajor, semiMinor, ellipseArea, axisRatio, angle, com, average, ellipseIntensity



# Input: Image (2D array format), binarization threshold (0-255)
# Output: [weighted_CenterOfMass, unweighted_CenterOfMass]
def findCenter(image, threshold):
    # Leaves anything larger than threshold as is
    max, imageBinaryWeighted = cv2.threshold(image, threshold, 1, cv2.THRESH_TOZERO)
    weightedCOM = ndimage.center_of_mass(imageBinaryWeighted)
    return weightedCOM


def convertMissingToZero(image):
    newImage = image.copy()
    indices = np.where(newImage == 255)
    if(len(indices[0]) == 0):
        return image
    for x in range(len(indices[0])):
        newImage[indices[0][x], indices[1][x]] = 0
    return newImage

    
# Input: Matrix of centerOfMasses as X and Y components (size Nx2)
# Output: Velocity Matrix as X and Y component vectors (size (N-1)x2) (Units = pixels/timestep where default timestep is 5 minutes)
def findCOMVelocity(coms):
    vel = np.diff(coms, axis=0)
    # vel = np.roll(vel, -1)
    return vel


# Input: Vector of Angles of length N
# Output: Vector of Angular (Units = degrees/timestep where default timestep is 5 minutes)
def findAngularVelocity(angles):
    vel = np.diff(angles, axis=0)
    # vel = np.roll(vel, -1)
    return vel



def findAverageIntensity(image):
    print(imageBinaryX.size)
    print(weightedImageBinaryX.size)
    avgInt = weightedImageBinaryX.size/imageBinaryX.size
    return 0


def checkMissingness(image):
    missingCount = np.count_nonzero(image == 255)
    # print('Percent Missing Data:', missingCount / image.size)
    percentMissing = missingCount/image.size
    return percentMissing


def read_data( sample_event, img_type, data_path=DATA_PATH ):
    """
    Reads single SEVIR event for a given image type.
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """
    fn = sample_event[sample_event.img_type==img_type].squeeze().file_name
    fi = sample_event[sample_event.img_type==img_type].squeeze().file_index
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data=hf[img_type][fi] 
    return data
