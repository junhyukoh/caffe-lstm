__author__ = 'davle_000'

import cv2
import numpy as np

def GetIntersectionRatio(patchBox, objectBox):
    # for thin objects
    if objectBox[0] > patchBox[0] and objectBox[2] < patchBox[2]:
        return 1.0
    elif objectBox[1] > patchBox[1] and objectBox[3] < patchBox[3]:
        return 1.0
    intersectionBox = [0] * 4
    intersectionBox[0] = max(patchBox[0], objectBox[0])
    intersectionBox[1] = max(patchBox[1], objectBox[1])
    intersectionBox[2] = min(patchBox[2], objectBox[2])
    intersectionBox[3] = min(patchBox[3], objectBox[3])
    if intersectionBox[0] > intersectionBox[2] or intersectionBox[1] > intersectionBox[3]:
        return 0
    patchBoxArea = (patchBox[2] - patchBox[0]) * (patchBox[3] - patchBox[1])
    intersectionBoxArea = (intersectionBox[2] - intersectionBox[0]) * (intersectionBox[3] - intersectionBox[1])
    return float(intersectionBoxArea) / patchBoxArea


def GenerateLabeledPatches(orignialImage, boundingBoxes, patchSize, startX, startY):
    X = [[], [], []]
    while startY + patchSize[0] < orignialImage.shape[0]:
        while startX + patchSize[1] < orignialImage.shape[1]:
            patchBox = [startY, startX, startY + patchSize[0], startX + patchSize[1]]
            maxIntersectionRatio = 0.0
            for boundingBox in boundingBoxes:
                intersectionRatio = GetIntersectionRatio(patchBox, boundingBox)
                maxIntersectionRatio = max(intersectionRatio, maxIntersectionRatio)
            if maxIntersectionRatio == 1.0:
                # inner case
                X[2].append(orignialImage[startY:startY + patchSize[0], startX:startX + patchSize[1]])
            elif maxIntersectionRatio == 0.0:
                # outer case
                X[0].append(orignialImage[startY:startY + patchSize[0], startX:startX + patchSize[1]])
            elif maxIntersectionRatio > 0.4 and maxIntersectionRatio < 0.6:
                X[1].append(orignialImage[startY:startY + patchSize[0], startX:startX + patchSize[1]])
            startX += patchSize[1]
        startY += patchSize[0]
        startX = 0
    return X

def ResizeBoundingBoxes(boundingBoxes, ratio):
    newBoundingBoxes = []
    for boundingBox in boundingBoxes:
        newBoundingBoxes.append([int(coord * ratio) for coord in boundingBox])
    return newBoundingBoxes

from xml.dom import minidom
import random
def ProcessOneFile(annotationFilePath, imageFilePath, minWidth, maxWidth, minHeight, maxHeight, patchSize, setSize):
    xmldoc = minidom.parse(annotationFilePath)
    originalWidth = int(xmldoc.getElementsByTagName('width')[0].firstChild.nodeValue)
    originalHeight = int(xmldoc.getElementsByTagName('height')[0].firstChild.nodeValue)
    boundingBoxes = []
    for i in range(len(xmldoc.getElementsByTagName('xmin'))):
        if xmldoc.getElementsByTagName('name')[i].firstChild.nodeValue != xmldoc.getElementsByTagName('folder')[0].firstChild.nodeValue:
            continue
        boundingBox = [0] * 4
        boundingBox[0] = int(xmldoc.getElementsByTagName('ymin')[i].firstChild.nodeValue)
        boundingBox[1] = int(xmldoc.getElementsByTagName('xmin')[i].firstChild.nodeValue)
        boundingBox[2] = int(xmldoc.getElementsByTagName('ymax')[i].firstChild.nodeValue)
        boundingBox[3] = int(xmldoc.getElementsByTagName('xmax')[i].firstChild.nodeValue)
        boundingBoxes.append(boundingBox)
    print(boundingBoxes)
    X = [[], [], []]
    img = cv2.imread(imageFilePath, cv2.IMREAD_COLOR)
    indicesToCheck = set([0, 1, 2])
    while True:
        shouldContinue = False
        for index in indicesToCheck:
            if len(X[index]) < setSize:
                shouldContinue = True
        if not shouldContinue:
            break
        newWidth = random.randint(minWidth, maxWidth)
        newHeight = random.randint(minHeight, maxHeight)
        ratio1 = float(newWidth) / originalWidth
        ratio2 = float(newHeight) / originalHeight
        ratio = min(ratio1, ratio2, 1.0)
        print( ratio )
        newBoundingBoxes = ResizeBoundingBoxes(boundingBoxes, ratio)
        newWidth = int(originalWidth * ratio)
        newHeight = int(originalHeight * ratio)
        startX = random.randint(0, patchSize[1])
        startY = random.randint(0, patchSize[0])
        print(startX, startY)
        print(newBoundingBoxes)
        newImg = cv2.resize(img, (newWidth, newHeight))
        dataset = GenerateLabeledPatches(newImg, newBoundingBoxes, patchSize, startX, startY)
        for i in range(len(X)):
            if len(X[i]) < setSize:
                X[i].extend(dataset[i])
            if len(X[i]) > setSize:
                X[i] = X[i][:setSize]
                for j in range(len(X)):
                    if len(X[j]) == 0 and j in indicesToCheck:
                        indicesToCheck.remove(j)
    return X
import sys
annotationsDir = sys.argv[1]
dataDir = sys.argv[2]
trainFiles = int(sys.argv[4])
print
#dataset = ProcessOneFile('n04152593_991.xml', 'n04152593_991.JPEG', 100, 300, 100, 300, (36, 36), 1000)
import os
trainDataSet = [[], [], []]
testDataSet = [[], [], []]
for iFile, annotationFile in enumerate(os.listdir(annotationsDir)):
    dataFile = os.path.join(dataDir, annotationFile[:-3] + 'JPEG')
    assert(os.path.exists(dataFile))
    annotationFile = os.path.join(annotationsDir, annotationFile)
    newData = ProcessOneFile(annotationFile, dataFile, 100, 700, 100, 700, (36, 36), 1000)
    if iFile < trainFiles:
        for i in range(len(trainDataSet)):
            trainDataSet[i].extend(newData[i])
    else:
        for i in range(len(testDataSet)):
            testDataSet[i].extend(newData[i])

with open(os.path.join(sys.argv[3],'train.txt'), 'w') as fout:
    for i in range(len(trainDataSet)):
        if not os.path.exists(os.path.join(sys.argv[3], 'train', str(i))):
            os.makedirs(os.path.join(sys.argv[3], 'train', str(i)))
        for j in range(len(trainDataSet[i])):
            cv2.imwrite(os.path.join(sys.argv[3], 'train', str(i), str(j) + '.png'), trainDataSet[i][j])
            fout.write('{0}/{1}.png {0}\n'.format(i, j))

with open(os.path.join(sys.argv[3],'test.txt'), 'w') as fout:
    for i in range(len(testDataSet)):
        if not os.path.exists(os.path.join(sys.argv[3], 'test', str(i))):
            os.makedirs(os.path.join(sys.argv[3], 'test', str(i)))
        for j in range(len(testDataSet[i])):
            cv2.imwrite(os.path.join(sys.argv[3], 'test', str(i), str(j) + '.png'), testDataSet[i][j])
            fout.write('{0}/{1}.png {0}\n'.format(i, j))