"""
This script initializes a dataset according to the user inputs. The user can choose one or more of the defects 
to be included in the dataset including the number of samples per defect. 

In order for the script to work correctly, the folder containing this file should have the following subfolders:

data
|__Annotations
|__Images
|__Images_bb
|__Pixel_masks

In order to not push the 'data' subfolder to the remote repo, the working directory should have the file '.gitignore' with the line '/data/'.

Furthermore, the parent directory of the working directory should contain the folders:

data_full
|__Annotations
|__Images

So the folder structure should look like this, where 'pcb_project' is the folder containing this python file:

pcb_project
|__data
|  |__Annotations
|  |__Images
|  |__Images_bb
|  |__Pixel_masks
|
data_full
|__Annotations
|__Images


The script then does the following:
1) it deletes the contents of the subfolders 'Annotations', 'Images', 'Images_bb' and 'Pixel_masks'
2) it copies a random choice of samples of the chosen defect types and respective size into the folder '/data/Images/'
3) it copies the according annotation xml files into the folder '/data/Annotations/'
4) it generates a corresponding csv file 'PCB_annotations_dataset.csv' in the working folder with one row for each defect instance, i.e. multiple rows per image
5) for each image in the dataset it generates an image with the drawn bounding boxes around the defects in the folder '/data/Images_bb/'
6) for each image in the dataset it generates an a pixel mask (the label) which is white on the defect locations and black otherwise in the folder '/data/Pixel_masks/'
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import cv2, os, re

# function definitions
def draw_bounding_boxes(df, filename, img_path):
    """
    draws the bounding box into the PCB image and saves it in the folder 'data/Images_bb'.

    returns: 
        the image with the bounding boxes (numpy.ndarray)
        the image file name (string)

    Args:
        df (pandas.DataFrame): a PCB-annotation DataFrame 
        filename (string): the filename of the PCB image with file type ending
        img_path (string): the relative path to the folder containing the image(s)
    """
    if filename in os.listdir(img_path):
        # read one image according to the path and filename
        img = cv2.imread(os.path.join(img_path,filename))
        # create a dataframe for each image from the PCB-annotation file with as many rows as there are defects in that image    
        df_grouped = df.groupby('filename')
        pcb = df_grouped.get_group(filename[:-4])
        # for each defect draw a red frame along the border of the bounding box
        for row in range(pcb.shape[0]):
            # along the vertical borders
            for j in range(pcb.ymin.iloc[row], pcb.ymax.iloc[row]+1):
                # line of width +/-5 around xmin and xmax
                for i in range(5):
                    # take max and min to handle the borders of the image
                    img[j][max(pcb.xmin.iloc[row]-i, 0)] = (0,0,255)
                    img[j][min(pcb.xmax.iloc[row]+i, 599)] = (0,0,255)
            for i in range(pcb.xmin.iloc[row], pcb.xmax.iloc[row]+1):
                # line of width +/-5 around ymin and ymax
                for j in range(5):
                    # take max and min to handle the borders of the image
                    img[max(pcb.ymin.iloc[row]-j, 0)][i] = (0,0,255)
                    img[min(pcb.ymax.iloc[row]+j, 599)][i] = (0,0,255)
        return(img, filename)
    else:
        print(f"Image {filename} not found in {img_path}")

def generate_pixel_matrix(df, filename):
    """
    generates a black image of the same shape as the passed PCB image, with white white pixels 
    exactly inside the defect bounding box(es) of the passed PCB image 
    returns:
        the pixel matrix (numpy.ndarray)
        filename without ending (string)

    Args:
        df (pandas.DataFrame): a PCB-annotation DataFrame 
        filename (string): the filename of the PCB image without file type ending
    """
    df_grouped = df.groupby('filename')
    # create a dataframe for each annotation file with as many rows as there are defects
    pcb = df_grouped.get_group(filename)
    # create a width x height marix of zeros, i.e. black pixels
    mask = np.zeros((pcb['width'].iloc[0], pcb['height'].iloc[0]))
    # for each defect set the pixels inside the retrieved bounding box to white
    for row in range(pcb.shape[0]):
        for i in range(pcb.ymin.iloc[row], pcb.ymax.iloc[row]+1):
            for j in range(pcb.xmin.iloc[row], pcb.xmax.iloc[row]+1):
                mask[i][j] = 255
    return(mask, filename)

def get_user_choice():
    defects = {1: 'missing_hole', 2: 'mouse_bite', 3: 'open_circuit', 4: 'short_', 5: 'spur_', 6: 'spurious_copper'}
    user_input = ''
    while not (re.compile(r"^(?!.*(\d).*\1)[1-6](?: [1-6](?!.*\1)){0,5}$").match(str.strip(user_input))): 
        user_input = input(f'Please select one or more defect types from:\n{defects}\n(separated by blank spaces, no duplicates):')
    chosen_defects = list(map(lambda x: defects[int(x)], str.split(str.strip(user_input), ' ')))
    user_input = ''
    while not (re.compile(r"\d{1,3}").match(str.strip(user_input))):
        user_input = input('How many images per defect (integer, max. 999)? ')
    chosen_size = int(str.strip(user_input))
    return(chosen_defects, chosen_size)

def clear_subfolders():
    # clearing subfolders 'Annotations', 'Images', 'Images_bb', 'Pixel_masks'
    for filename in os.listdir(image_dest_path):
        os.remove(os.path.join(image_dest_path,filename))
    for filename in os.listdir(annot_dest_path):
        os.remove(os.path.join(annot_dest_path,filename))
    for filename in os.listdir(bb_path):
        os.remove(os.path.join(bb_path, filename))
    for filename in os.listdir(mask_path):
        os.remove(os.path.join(mask_path, filename))

def copy_samples(chosen_defects, chosen_size):
    # selecting only the images with the chosen defects
    pool = {}
    for defect_name in chosen_defects:
        pool[defect_name]=[]
        for filename in os.listdir(image_pool_path):        
            if defect_name in filename:
                pool[defect_name].append(filename)
    print(f"Picking from {chosen_defects}")

    for defect in pool.keys():
        rnd_picks = np.random.choice(pool[defect], min(len(pool[defect]), chosen_size), replace=False)
        for filename in rnd_picks:
            os.system(f"cp {image_pool_path}{filename} {image_dest_path}")
            os.system(f"cp {annot_pool_path}{filename[:-4]}.xml {annot_dest_path}")

    # correct filename tags in xml files to coincide with file name
    for filename in os.listdir(annotation_path):
        tree = ET.parse(os.path.join(annotation_path, filename))
        for node in tree.iter():
            if node.tag == 'filename':
                node.text = filename[:-4]
        tree.write(annotation_path+filename)

def generate_PCB_csv():
    dataset = {
    'filename': [],
    'width': [],
    'height': [],
    'depth': [],
    'defect': [],
    'xmin': [],
    'xmax': [],
    'ymin': [],
    'ymax': []
    }

    # iterate over all files in folder
    for filename in os.listdir(annotation_path):
        tree = ET.parse(os.path.join(annotation_path, filename))
        # create a row for each 'object' Element, i.e. for each single defect
        for obj in [node for node in list(tree.iter()) if node.tag == 'object']:
            for node in obj:
                if node.tag == 'name':
                    dataset['defect'] += [node.text]
                if node.tag == 'bndbox':
                    for child in node:
                        # use a regular expression to match all bounding element tags
                        if re.compile(r'^(x|y)(min|max)').match(child.tag):
                            dataset[child.tag] += [int(child.text)]    
            # writing the data which is identical for each occurence of 'object' in one file
            for node in tree.iter():
                # each filename can appear more than once if the PCB has more than one defect
                # so it is not viable as row ID
                if node.tag == 'filename':
                    dataset[node.tag] += [node.text]
                if node.tag in ['width', 'height', 'depth']:
                    dataset[node.tag] += [int(node.text)]
                    
    df = pd.DataFrame(dataset) 
    df.to_csv(f'{working_path}PCB_annotations_dataset.csv', sep=';', index=False)
    return(df)

# set up folder paths
working_path = os.path.dirname(os.path.abspath(__file__)) + '/'
image_pool_path = working_path + '../data_full/Images/all/'
image_dest_path = working_path + 'data/Images/'
annot_pool_path = working_path + '../data_full/Annotations/'
annot_dest_path = working_path + 'data/Annotations/'
img_path = image_dest_path # only for more intuitive variable names later on
annotation_path = annot_dest_path # only for more intuitive variable names later on
bb_path = working_path + 'data/Images_bb/'
mask_path = working_path + 'data/Pixel_masks/'

clear_subfolders()
chosen_defects, chosen_size = get_user_choice()
copy_samples(chosen_defects, chosen_size)
df = generate_PCB_csv()

# call the function draw_bounding_boxes once for each image
for filename in os.listdir(img_path):
    img, filename = draw_bounding_boxes(df, filename, img_path)
    cv2.imwrite(f"{bb_path}bb-{filename}", img)

# call the function generate_pixel_matrix once for each PCB in the annotation file
for filename in df.filename.unique():
    mask, filename = generate_pixel_matrix(df, filename)
    cv2.imwrite(f"{mask_path}/pm-{filename}.png", mask)