
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET 
import warnings
warnings.filterwarnings("ignore")

#path where the Annotations xml files are stored within the extracted dataset folder
path_annotations = "C:/Users/faiza/Downloads/VOC_PCB/VOC_PCB/Annotations/"

#identifying important properties/attributes of the document tree in the xml file
#ignoring attributes such as 'source', 'owner', segmented', 'pose', 'truncated', etc.
dataset = {
            "file":[],
            "width":[],
            "height":[],
            "depth":[],
            "class":[],
            "xmin":[],
            "ymin":[],   
            "xmax":[],
            "ymax":[],
           }

all_files = []

for path, subdirs, files in os.walk(path_annotations):
     for name in files:
        all_files.append(os.path.join(path, name))

#augmenting the dataset
for annotations in all_files:
    tree = ET.parse(annotations)
    for element in tree.iter():
        if 'size' in element.tag:
            for attribute in list(element):
                if 'width' in attribute.tag: 
                    width = int(round(float(attribute.text)))
           
                if 'height' in attribute.tag:
                    height = int(round(float(attribute.text)))  
                    
                if 'depth' in attribute.tag:
                    depth = int(round(float(attribute.text)))    

        if 'object' in element.tag:
            # print('[object] in element.tag ==> list(elem)\n'), print(list(element))
            for attribute in list(element):
                
                # print('attr = %s\n' % attr)
                if 'name' in attribute.tag:
                    name = attribute.text                 
                    dataset['class']+=[name]
                    dataset['width']+=[width]
                    dataset['height']+=[height] 
                    dataset['depth']+=[depth] 
                    dataset['file']+=[annotations.split('/')[-1][0:-4]] 
                            
                if 'bndbox' in attribute.tag:
                    for dimensions in list(attribute):
                        if 'xmin' in dimensions.tag:
                            xmin = int(round(float(dimensions.text)))
                            dataset['xmin']+=[xmin]
                        if 'ymin' in dimensions.tag:
                            ymin = int(round(float(dimensions.text)))
                            dataset['ymin']+=[ymin]                                
                        if 'xmax' in dimensions.tag:
                            xmax = int(round(float(dimensions.text)))
                            dataset['xmax']+=[xmax]                                
                        if 'ymax' in dimensions.tag:
                            ymax = int(round(float(dimensions.text)))
                            dataset['ymax']+=[ymax]     


df=pd.DataFrame(dataset)
#print(df.head(10))

df.index.name = 'Index'

df.to_csv("PCB_annotations_dataset.csv", sep=';')
