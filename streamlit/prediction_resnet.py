import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Recall, Precision, MeanAbsoluteError, MeanIoU

current_script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_script_directory)

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Define the destination path for images and annotations if needed
image_dest_path = os.path.join(parent_dir, 'data', 'Images')
annot_dest_path = os.path.join(parent_dir, 'data', 'Annotations')

# Define the path for RES_UNET model
model_loc = os.path.join(parent_dir, 'notebooks', 'models')

# Load the model
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
        #image_test = cv2.resize(image_test, (600, 600))
        #image = Image.open(test_img)
        #image_test = np.array(image)

        if image_test is not None:
            crop_size = (100,100)
            #print(f"Original image size: {image_test.shape[:2]}")
            height, width = image_test.shape[:2]
            max_dimension = 1200
            check = max(height,width)
            if check > 3000:
                max_dimension = 1900
            #elif 3000 >= check > 2000 :
            #    max_dimension = 1200
            scaling_factor = min(max_dimension / height, max_dimension / width)
            if scaling_factor < 1:
                new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
                image_test = cv2.resize(image_test, new_dimensions, interpolation=cv2.INTER_AREA)

            height, width = image_test.shape[:2]

            image_yuv = cv2.cvtColor(image_test, cv2.COLOR_BGR2YUV)
            avg_brightness = np.mean(image_yuv[:, :, 0])
            print("avg_brightness 1:", avg_brightness)
            brightness_threshold = 65
            if avg_brightness < brightness_threshold:
                cliplimit = round(125 / avg_brightness, 1) #+ 0.2
                print(cliplimit)
                clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(10, 10))
                image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
                avg_brightness = np.mean(image_yuv[:, :, 0])
                print("avg_brightness 2:", avg_brightness)

            image_test = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

            target_height = round((height + 25) / 100) * 100
            target_width = round((width + 25) / 100) * 100
    
            top_crop = max(0, (height - target_height) // 2)
            bottom_crop = top_crop + target_height
            left_crop = max(0, (width - target_width) // 2)
            right_crop = left_crop + target_width
    
            cropped_image = image_test[top_crop:bottom_crop, left_crop:right_crop]
            #cropped_image = np.array(cropped_image)
            pad_left = 0
            pad_top = 0
            if cropped_image.shape[0] != target_height or cropped_image.shape[1] != target_width:
                pad_top = max(0, (target_height - cropped_image.shape[0]) // 2)
                pad_bottom = max(0, target_height - cropped_image.shape[0] - pad_top)
                pad_left = max(0, (target_width - cropped_image.shape[1]) // 2)
                pad_right = max(0, target_width - cropped_image.shape[1] - pad_left)
                
                cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        
            print(f"Cropped and padded image size: {cropped_image.shape[:2]}")
            
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

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
            
            total_area = target_height * target_width
            if total_area > 2100000:
                min_contour_area = total_area * 0.0001
                max_contour_area = total_area * 0.00035
                #min_contour_area = 250
                #max_contour_area = 600
            elif 2000000 < total_area <= 2100000:
                min_contour_area = total_area * 0.00010
                max_contour_area = total_area * 0.00055
            elif 1200000 < total_area <= 2000000:
                if max_dimension > 1500:
                    min_contour_area = total_area * 0.00023
                    max_contour_area = total_area * 0.0008
                else:
                    min_contour_area = total_area * 0.00007
                    max_contour_area = total_area * 0.0015
            elif 800000 < total_area <= 1200000:
                min_contour_area = total_area * 0.00027
                max_contour_area = total_area * 0.0018 #0.0007
            elif 400000 < total_area <= 800000:
                min_contour_area = total_area * 0.00045  # was 0.0005
                max_contour_area = total_area * 0.0022
            elif 200000 < total_area <= 400000:
                min_contour_area = total_area * 0.0006
                max_contour_area = total_area * 0.0064
            elif 100000 < total_area <= 200000:
                min_contour_area = total_area * 0.0013
                max_contour_area = total_area * 0.0030
            else:  # total_area <= 100000
                min_contour_area = total_area * 0.0015
                max_contour_area = total_area * 0.00165

            if avg_brightness < 70:
                min_contour_area = total_area * 0.00005
                max_contour_area = total_area * 0.00165
            #elif avg_brightness > 100:
             #   min_contour_area = 200
              #  max_contour_area = 1500

            print(total_area)
            print(min_contour_area)
            print(max_contour_area)
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

            base_scale = min(width, height) / 1000.0
            font_scale = base_scale   
            thickness = int(base_scale * 4)  

            for contour, bbox, label, conf in zip(filtered_contours, bounding_boxes, remapped_pred_label, y_pred_confidence):
                if label[0] == "none" :#or conf < 0.5:
                    continue  # skip contour if the class label is "none"    
                x, y, w, h = bbox
                class_label = f"{label[0]} {conf[0]:.2f}"
                num_defect = num_defect + 1
                margin = 10 
                x_new = x - (margin + pad_left)
                y_new = y - (margin + pad_top)
                w_new = w + margin
                h_new = h + (margin+5)

                cv2.rectangle(output_image, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 0), 2)

                text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_bg_size = (text_size[0] + 10, text_size[1] + 10)

                cv2.rectangle(output_image, (x_new, y_new + h_new), (x_new + text_bg_size[0], y_new + h_new + text_bg_size[1]), (255, 0, 0), -1) 

                # putting text (class label) below the bounding box
                cv2.putText(output_image, str(class_label), (x_new, y_new + h_new + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


            return output_image, num_defect

        else:
            print(f"Unable to read image from {test_img}")
    else:
        print("No images found in the directory.")
