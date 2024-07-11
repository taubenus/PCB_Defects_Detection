import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.metrics import Recall, Precision, MeanAbsoluteError, MeanIoU

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Define the destination path for images and annotations if needed
image_dest_path = os.path.join(parent_dir, 'data', 'Images')
annot_dest_path = os.path.join(parent_dir, 'data', 'Annotations')

# Define the path for RES_UNET model
model_loc = os.path.join(parent_dir, 'notebooks', 'models')

# Load the model
from tensorflow.keras.models import load_model
model_path = os.path.join(model_loc, 'model_enhanced_res_unet-v240618_3.keras')

model = load_model(model_path, custom_objects={
    'MeanAbsoluteError': MeanAbsoluteError,
    'MeanIoU': MeanIoU,
    'Recall': Recall,
    'Precision': Precision
})

class_labels_pred = {
    0: 'missing_hole',
    1: 'mouse_bite',
    2: 'none',
    3: 'open_circuit',
    4: 'short',
    5: 'spur',
    6: 'spurious_copper'
}

def crop_image(image, crop_size=(100, 100)):
    crops = []
    #print(image.shape)
    height, width = image.shape
    crop_height, crop_width = crop_size

    for i in range(0, height, crop_height):
        for j in range(0, width, crop_width):
            crop = image[i:i + crop_height, j:j + crop_width]
            if crop.shape[0] == crop_height and crop.shape[1] == crop_width:
                crops.append(crop)
    return crops


def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred = np.array(blurred, dtype=np.uint8)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = np.array(thresh, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thresh
	
def detect_and_filter_contours(binary_image, min_contour_area, max_contour_area):
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if (max_contour_area > cv2.contourArea(contour) > min_contour_area)]
    
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        print(area)
    bounding_boxes = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    return filtered_contours, bounding_boxes
	
def draw_contours(original, image, contours):
    #print(original.shape)
    #print(image.shape)
    if len(original.shape) == 2 or original.shape[2] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #print(x, y, w, h)
        cv2.drawContours(output_image, [contour], -1, 255, -1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), 255, -1)

    alpha = 0.6  
    output_image = cv2.addWeighted(original, 1, output_image, alpha, 0)
    return output_image
	
	
def combine_crops(crops, crop_size, full_size):
    rows = full_size[0] // crop_size[0]
    cols = full_size[1] // crop_size[1]
    combined_image = np.zeros(full_size)

    crop_idx = 0
    for i in range(rows):
        for j in range(cols):
            combined_image[i*crop_size[0]:(i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]] = crops[crop_idx]
            crop_idx += 1
    return combined_image
	
def crop_and_resize_images(image, bounding_boxes, size=(100, 100)):
    cropped_images = []
    
    for bbox in bounding_boxes:
        #print("Bounding Box:", bbox)
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        #print("Center:", cx, cy)
        start_x = max(0, cx - size[0] // 2)
        start_y = max(0, cy - size[1] // 2)
        #print("Start Point:", start_x, start_y)
        start_x = min(start_x, image.shape[1] - size[0])
        start_y = min(start_y, image.shape[0] - size[1])
        #print("Adjusted Start Point:", start_x, start_y)
        end_x = start_x + size[0]
        end_y = start_y + size[1]
        #print("End Point:", end_x, end_y)
        cropped = image[start_y:end_y, start_x:end_x]
        #print("Cropped Shape:", cropped.shape)
        cropped_resized = cv2.resize(cropped, size, interpolation=cv2.INTER_LINEAR)
        #print("Resized Shape:", cropped_resized.shape)
        cropped_images.append(cropped_resized)
    return cropped_images

def process_image(num, test_img):
    if test_img:
        
        image_test = cv2.imread(test_img)
        image_test = cv2.resize(image_test, (600, 600))

        if image_test is not None:
            crop_size = (100,100)
            #print(f"Original image size: {image_test.shape[:2]}")
            height, width = image_test.shape[:2]
            gray_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
            image_array_test = np.asarray(gray_image) / 255.0
            crops = np.asarray(crop_image(image_array_test, crop_size))
            y_pred, y_class = model.predict(crops)
            predicted_label = np.argmax(y_class, axis=1)
            remapped_label = [class_labels_pred[label] for label in predicted_label]

            X_test_combined = combine_crops(crops, crop_size, cropped_image.shape[:2])
            y_mask_pred_combined = combine_crops(y_pred.squeeze(), crop_size, cropped_image.shape[:2])

            normalized_image = cv2.normalize(y_mask_pred_combined, 5, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normalized_image = np.uint8(normalized_image)

            # preprocess the image
            thresh = preprocess_image(normalized_image)

            min_contour_area = 150  
            max_contour_area = 1000  
  
            filtered_contours, bounding_boxes = detect_and_filter_contours(thresh, min_contour_area, max_contour_area)

            output_image = draw_contours(cropped_image, normalized_image, filtered_contours)

            if num == 1:
                return output_image, len(filtered_contours)

            #print("Number of contours found:", len(filtered_contours))

            cropped_images = crop_and_resize_images(X_test_combined, bounding_boxes)

            num_images = len(cropped_images)

            y_mask_pred = []
            y_class_pred = []
            pred_label = []
            remapped_pred_label = []
            y_pred_confidence = []

            for i in range(num_images):
                cropped_image = cropped_images[i]
                cropped_image = np.expand_dims(cropped_image, axis=0)  
                #cropped_images_batch = np.expand_dims(cropped_images[i], axis=0)
                #print(cropped_images_batch.shape)
                y_mask_pred_i, y_class_pred_i = model.predict(cropped_image)

                y_mask_pred.append(y_mask_pred_i)
                y_class_pred.append(y_class_pred_i)

                pred_label_i = np.argmax(y_class_pred_i, axis=1)
                remapped_pred_label_i = [class_labels_pred[label] for label in pred_label_i]

                y_pred_confidence_i = np.asarray(list(map(lambda cat: np.max(cat), y_class_pred_i)))   

                pred_label.append(pred_label_i)
                remapped_pred_label.append(remapped_pred_label_i)
                y_pred_confidence.append(y_pred_confidence_i)

            normalized_image = cv2.normalize(y_mask_pred_combined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normalized_image = np.uint8(normalized_image)
            thresh = preprocess_image(normalized_image)
            filtered_contours, bounding_boxes = detect_and_filter_contours(thresh, min_contour_area, max_contour_area)

            output_image = image_test.copy()  
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  
            num_defect = 0

            font_scale = 1   
            thickness = 2

            for contour, bbox, label, conf in zip(filtered_contours, bounding_boxes, remapped_pred_label, y_pred_confidence):
                if label[0] == "none" :#or conf < 0.5:
                    continue  # skip contour if the class label is "none"    
                x, y, w, h = bbox
                class_label = f"{label[0]} {conf[0]:.2f}"
                num_defect = num_defect + 1
                margin = 10 
    
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_bg_size = (text_size[0] + 10, text_size[1] + 10)
                cv2.rectangle(output_image, (x, y + h), (x_new + text_bg_size[0], y + h + text_bg_size[1]), (255, 0, 0), -1) 

                # putting text (class label) below the bounding box
                cv2.putText(output_image, str(class_label), (x, y + h + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            return output_image, num_defect

        else:
            print(f"Unable to read image from {test_img}")
    else:
        print("No images found in the directory.")
