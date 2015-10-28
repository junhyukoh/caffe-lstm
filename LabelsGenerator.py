__author__ = 'davle_000'

import sys
root_folder = sys.argv[1] #"C:/Users/davle_000/PycharmProjects/ImageNet"
import os
annotations_train = os.path.join(root_folder, "Annotations/CLS-LOC/train")
annotations_val = os.path.join(root_folder, "Annotations/CLS-LOC/val")
data_train = os.path.join(root_folder, "Data/CLS-LOC/train")
data_val = os.path.join(root_folder, "Data/CLS-LOC/val")

train_lines = []
val_lines = []
labels = {}
for dir in os.listdir(annotations_train):
    labels[dir] = len(labels)

from xml.dom import minidom
def GetLine(annotation_file, annotation_dir, data_dir):
    xmldoc = minidom.parse(os.path.join(root_folder, annotation_dir, annotation_file))
    lines = []
    for i in range(len(xmldoc.getElementsByTagName('xmin'))):
        label_str = xmldoc.getElementsByTagName('name')[i].firstChild.nodeValue
        if label_str not in labels:
            continue
        boundingBox = [0] * 4
        boundingBox[0] = int(xmldoc.getElementsByTagName('ymin')[i].firstChild.nodeValue)
        boundingBox[1] = int(xmldoc.getElementsByTagName('xmin')[i].firstChild.nodeValue)
        boundingBox[2] = int(xmldoc.getElementsByTagName('ymax')[i].firstChild.nodeValue)
        boundingBox[3] = int(xmldoc.getElementsByTagName('xmax')[i].firstChild.nodeValue)
        img_path = os.path.join(root_folder, data_dir, annotation_file[:-3] + 'JPEG')
        assert(os.path.exists(img_path))
        line = os.path.join(data_dir, annotation_file[:-3] + 'JPEG') + " " + str(labels[label_str])
        for j in range(len(boundingBox)):
            line += " " + str(boundingBox[j])
        lines.append(line)
    return lines

for label_str, label in labels.items():
    for annotation_file in os.listdir(os.path.join(annotations_train, label_str)):
        lines = GetLine(annotation_file, "Annotations/CLS-LOC/train/" + label_str, "Data/CLS-LOC/train/" + label_str)
        train_lines.extend(lines)
for annotation_file in os.listdir(annotations_val):
    lines = GetLine(annotation_file, "Annotations/CLS-LOC/val", "Data/CLS-LOC/val")
    val_lines.extend(lines)

with open('train.txt', 'w') as fout:
    for line in train_lines:
        fout.write(line)
        fout.write("\n")
with open('val.txt', 'w') as fout:
    for line in val_lines:
        fout.write(line)
        fout.write("\n")