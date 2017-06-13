import cv2
import numpy as np
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt

class TrainingEx:
    __matrix = None
    __score = None

    def __init__(self, matrix, score):
        self.__matrix = matrix
        self.__score = score

    def get_matrix(self):
        return self.__matrix

    def get_score(self):
        return self.__score

class Ellipse:
    faces = []
    name = None

    def __init__(self, name, faces):
        self.name = name
        self.faces = faces

def readPictureNames():
    nameList = []
    with open('FDDB-folds/FDDB-fold-01.txt', 'r') as f:
        while len(nameList) < 5:
            n = f.readline().strip()
            nameList.append(n)
    return nameList

def readEllipses():
    ellipsesList = []
    with open('FDDB-folds/FDDB-fold-01-ellipseList.txt', 'r') as f:
        while len(ellipsesList) < 5:
            name = f.readline().strip()
            n = f.readline()
            n = int(n)
            faces = []
            for x in range(0, n):
                line = f.readline().strip()
                lst = line.split(' ')
                lst2 = [float(i) for i in lst]
                faces.append(lst2)
            ellipsesList.append(Ellipse(name, faces))
    return ellipsesList

def readImages():
    imagesList = []
    for i in range(0,5):
        imagesList.append(cv2.imread('originalPics.tar/' + name[i] + '.jpg', 1))
    return imagesList

def plot5(array):
    for i in range(0,5):
        cv2.imshow('image', array[i])
        cv2.waitKey(0)
    return

def plot5_plt(array):
    for i in range(0,5):
        plt.subplot(2,3,i+1)
        plt.imshow(array[i], cmap='gray')
    plt.show()

def convertToGray():
    imagesList = []
    for i in range(0,5):
        imagesList.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
    return imagesList

def createGaussian(sigma):
    matrix = np.zeros([6*sigma+1,6*sigma+1])
    filterRadius = sigma * 3
    for row in range(-filterRadius, filterRadius+1):
        for col in range(-filterRadius, filterRadius+1):
            matrix[row+filterRadius][col+filterRadius] = (1/(2*np.pi*sigma**2)) * np.exp(-(row**2 + col**2)/(2*sigma**2))
    return matrix

def convolveGaussian(sigma):
    imagesList = []
    gaussian = createGaussian(sigma)
    for i in range (0,5):
        imagesList.append(signal.convolve2d(images_grayscale[i], gaussian))
    return imagesList

def compressImages(factor):
    imagesList = []
    for i in range(0,5):
        [a, b] = images_gaussian[i].shape
        #imagesList.append(signal.decimate(signal.decimate(images_gaussian[i], factor, axis = 0),factor))
        imagesList.append(cv2.resize(images_gaussian[i], (0,0), fx=0.5, fy=0.5))
    return imagesList

def computeScore(image, center ,faceellipse):
    intercept = 0
    ccy, ccx = center
    [ra, rb, theta, cx, cy, dummy] = faceellipse
    for j in range(-16, 16):
        for i in range(-16, 16):
            x = ccx + i
            y = ccy + j
            dist = (((np.cos(theta)*(x-cx)+np.sin(theta)*(y-cy))**2)/ra**2) + (((np.sin(theta)*(x-cx)+np.cos(theta)*(y-cy))**2)/rb**2)
            if dist <= 1:
                intercept += 1
    score = intercept/((32**2)+(np.pi*ra*rb)-intercept)
    return score

def addFaces(img, faceellipse):
    [cx, cy] = faceellipse[3:5]
    [cx, cy] = [int(cx), int(cy)]
    for m in [-3, -2, -1, 0, 1, 2, 3]:
        for n in [-3, -2, -1, 0, 1, 2, 3]:
            center = [cy+2*m, cx+2*n]
            score = computeScore(img, center, faceellipse)
            trainingexamples.append(TrainingEx(img[cy + 2 * m - 16:cy + 2 * m + 16, cx + 2 * n - 16:cx + 2 * n + 16], score))

def resizeEllipse(picNum,faceNum):
    ellipses[picNum].faces[faceNum][0] = ellipses[picNum].faces[faceNum][0] / (2 * 2 ** (0.5))
    ellipses[picNum].faces[faceNum][1] = ellipses[picNum].faces[faceNum][1] / (2 * 2 ** (0.5))
    ellipses[picNum].faces[faceNum][3] = ellipses[picNum].faces[faceNum][3] / (2)
    ellipses[picNum].faces[faceNum][4] = ellipses[picNum].faces[faceNum][4] / (2)

def makeTrainingExamples():
    for picNum in range(0, len(images)):
        for faceNum in range(0, len(ellipses[picNum].faces)):
            if max(ellipses[picNum].faces[faceNum][0:2]) < 20:
                #normal face extract
                addFaces(images_gaussian[picNum], ellipses[picNum].faces[faceNum])
            elif max(ellipses[picNum].faces[faceNum][0:2]) < 110:
                #big face extract
                resizeEllipse(picNum,faceNum)
                addFaces(images_decimated[picNum], ellipses[picNum].faces[faceNum])


#READING PICTURE NAMES
name = readPictureNames()

#Reading ELLIPSES
ellipses = readEllipses()

#reading images
images = readImages()

#plot images
plot5(images)

#change images to grayscale
images_grayscale = convertToGray()

#gaussian blur & decimate
factor = 2
sigma = 3
images_gaussian = convolveGaussian(sigma)
images_decimated = compressImages(factor)

#plot images after decimated
plot5_plt(images_decimated)

#extracting faces
trainingexamples = []
makeTrainingExamples()

i=0
while i < len(trainingexamples)-1:
        plt.subplot(1, 3, 1)
        plt.title(trainingexamples[i].get_score())
        plt.imshow(trainingexamples[i].get_matrix(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title(trainingexamples[i+15].get_score())
        plt.imshow(trainingexamples[i+15].get_matrix(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title(trainingexamples[i+30].get_score())
        plt.imshow(trainingexamples[i+30].get_matrix(), cmap='gray')
        plt.show()
        i += 49
