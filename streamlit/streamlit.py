import streamlit as st
import os
import cv2
from PIL import Image
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report
from collections import Counter

@st.cache_data
def load_image(imageName):
    image = Image.open(os.getcwd() + '/figures/' + imageName)
    return image

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

image_path = os.path.abspath(os.path.join(parent_dir, 'data', 'Images_bb'))

st.sidebar.title("Table of contents")
pages = ["Project Introduction", "Data Exploration", "Feature Engineering", "Model Training", 
         "Model Optimization and Evaluation", 
         "Authors"]
page = st.sidebar.radio("Go to", pages)

#st.sidebar.header(pages[5])
st.sidebar.markdown("[*Faiza Waheed*](https://github.com/wfaiza/)")
st.sidebar.markdown("[*Niels Hartano*](https://github.com/taubenus/)")
st.sidebar.markdown("[*Gernot Gellwitz*](https://github.com/Kathartikon/)")


if page == pages[0]:
    st.write("# Detection and Classification of Defects on Printed Circuit Boards with Machine Learning")
    st.write("## Defect detection on PCBs (Segmentation and Classification)")
    st.write("""This project explores various machine learning methodologies for detecting and classifying 
defects on printed circuit boards (PCBs), using advanced computer vision techniques. Printed Circuit Boards 
(PCBs) are essential components in nearly all electronic devices. Ensuring their quality is critical, 
as defects can lead to device malfunctions or failures. Traditional manual inspection methods are 
time-consuming and error-prone, motivating the adoption of deep learning models such as VGG16, RES-UNET, 
and YOLOv5 for automated defect detection.""")
    st.write("""Our main object was to learn the architecture designing, training and deployment
of a manually designed RES_UNET model.""")
    st.write("#### Project Phases:")
    st.write("""The project encompassed several key stages of a rigorous data science methodology. 
We will cover these aspects of our Data Science project in detail, including:""")
   
    st.write("##### 1. Data Exploration:")
    st.write( """- Explore the dataset to understand its structure, features, and potential pitfalls.\n
- Use data visualization to identify key insights and relevance.\n
- Ensure the quality of the images. """)
       
    st.write("##### 2. Feature Engineering:")
    st.write("""- Perform feature engineering to balance and augment the data.\n
- Create the masks (segmentation) and the target labels (classification) for training images.\n
- Ensure the target labels were one-hot-encoded as per requirement of classification.\n
- Implement a randomization strategy to neutralize the impact of baises and ensure model impartiality.\n
- Ensuring the dataset is ready for model training.""")
    
    st.write("##### 3. Model Training:")
    st.write("""- Develop various machine learning models to detect the anomalies (defects).""")
    
    st.write("##### 4. Model Optimization and Evaluation:")
    st.write("""- Refine our designed model to maximize accuracy and robustness.
- Rigorous evaluation of model performance. """)
      
    st.write("""This PCB defect detection project has been a dedicated effort for us, blending rigorous data 
analysis and advanced feature engineering with the practical application of machine learning. 
Our goal was to excel in our endeavor by utilizing all available tools to successfully identify 
and classify PCB defects.""")
    

    image_1 = load_image('PCB-Final-Image.jpg')
    st.image(image_1, caption="Typical 2 layer PCB", use_column_width='auto')
    
elif page == pages[1]:
    st.write("# Data Exploration:")
    st.write("A sample image for defective PCB:")
    image_2 = load_image('missing_hole_1.jpg')
    st.image(image_2, caption="Sample of a defected PCB", width=500)

    st.write("##### The images are too large to handle without any pre-processing. Example Image dimensions: ", 
             image_2.size)
    st.write("This image dataset has over 10,000 synthetically generated images.")
    st.write("""- The dataset is located at:
             (https://www.kaggle.com/datasets/akhatova/pcb-defects)""")
    
    st.write("##### What can we say about the defect types?")
    st.write("""Let's look at the type of defects we will detect in this project: \n
- Missing hole\n
- Mouse bite\n
- Open circuit\n
- Short\n
- Spur\n
- Spurious copper""")
    with st.expander('View sample defect types', expanded=False):
        image_3 = load_image('Defect_types.png')
        st.image(image_3, caption="Sample defects explored in this project", use_column_width='auto')

    options = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur_', 'spurious_copper']
    st.markdown('### Sample images with defects')
    choice = st.selectbox('Select Defect', options, index=None, placeholder='Choose a defect type', label_visibility='collapsed')
    if choice is not None:
        img_pool_choice = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if choice in filename]
        rnd_3 = np.random.choice(range(20), 3, replace=False)
        fig = plt.figure(figsize=(45, 15))
        for i, j in enumerate(rnd_3):
            img = cv2.imread(img_pool_choice[j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 1, i + 1)
            plt.axis('off')
            plt.title(f"{choice} {i + 1}")
            plt.imshow(img)
        st.pyplot(fig, use_container_width=False)
		
		
    st.write("##### Is there a possibilty to minimize the features?")
    st.write("""Let's inspect the images from the dataset: \n""")

    image_4 = load_image('Annotated_defects.jpg')
    st.image(image_4, caption="Sample image from the PCB dataset with annotated defects", 
             width=400)
    
    # sample checkbox for superimposed images
    with st.expander('View superimposed dataset images', expanded=False):
        image_4a = load_image("superimposed_image.png")
        st.image(image_4a, caption='All the training images superimposed together')

    st.write("""After inspecting the different images, there doesnt seem to be a possibility of reducing 
features, as this will result in essential features being lost. All the various nuances 
of the image need to be preserved for an accurate and robust anomaly detection model.""")

elif page == pages[2]:
    st.write("# Feature Engineering")
    st.write("##### 1. Visualization of dataset")
    st.write("""We observed from the sample images, that the number of defects on each image varies. 
             Hence we looked at the distribution of the dataset, whether it was balanced or not.""")
    image_5 = load_image('data_balance.png')
    st.image(image_5, caption="Defect distribution", use_column_width='auto')
    
    st.write(""" **Observations**:  
             
- The dataset is relatively balanced overall. 
- Unfortunately this is the visualization before doing feature engineering to ensure model robustness. 
- We needed to augment the data and then reavaluate after implementing feature engineering.""")
    st.write("##### 2. Data Augmentation")
    st.write(
"""Data preprocessing plays a crucial role in constructing effective Machine Learning models, 
as the quality of prediction results is closely tied to the thoroughness of data preprocessing.
Our image preprocessing pipeline involved several key steps:\n\n""")
    st.write(
"""- **Dimension Handling:** Reducing the image dimensions initially from RGB to Grayscale,
and then cropping the the image to 100 x 100 grayscale images.\n""")
        
    image_6 = load_image('rgb vs grayscale.png')
    st.image(image_6, caption="Colored vs. Grayscale image", use_column_width='auto')

    st.write(
"""\n\n- **Mask and target label:** The reference mask for the defects was generated with the help of 
the detailed annotations provided along with the image dataset. Each bounding box coordinates were mapped onto 
the mask image and stored in memory along with the reference training image.\n
Besides this, the reference target labels were also generated for classification and one-hot encoded.
We have 6 types of defects in our project.\n\n""")

    st.write(
"""- **Augmentation:** We implemented data augmentation with the help of Albumentations library.
This unfortunatly did not cater for the the defects in the optimum way as some defects 
on the border would be cropped out. Nor did it cater for the instances where multiple defects were
located in one image. Hence we had to implement manual augmentation which included: \n\n""")
    
    image_7 = load_image('albumentations.png')
    st.image(image_7, caption="Possible augmentations by ”Albumentations", use_column_width='auto')
 
    image_8 = load_image('croppedimagewithmask.png')
    st.image(image_8, caption="Cropping image and mask to 100x100 dimension", use_column_width='auto')

    image_9 = load_image('manual augmentation.png')
    st.image(image_9, caption="Manually implemented augmentations", use_column_width='auto')

    st.write(
"""- ***Ensuring quality of defects:*** To keep the defects in all the cropped images intact, 
we had to implement a check function for border and defect control. This included a function to ensure 
that there were only single defects in one image, i.e. separation of images by duplication.""")

    new_width = 400
    image_10 = load_image('separated_img1.png')
    image_10 = image_10.resize((new_width, int((new_width / image_10.width) * image_10.height)))
    st.image(image_10, use_column_width='auto')
    image_11 = load_image('separated_img2.png')
    image_11 = image_10.resize((new_width, int((new_width / image_11.width) * image_11.height)))
    st.image(image_11, use_column_width='auto')
    image_12 = load_image('separated_img3.png')
    image_12 = image_12.resize((new_width, int((new_width / image_12.width) * image_12.height)))
    st.image(image_12, caption="Manually implemented defect separation", use_column_width='auto')

    st.write(
"""- **Randomization of dataset:** We implemented random; but defect class-wise balanced; 
dataset selection so that we could avoid traiing baises.""")

    st.write(
"""Once we have managed to implement the above mentioned feature engineering aspects, we can 
get on with the next step of Model architecture design and training.""")
    
elif page == pages[3]:

    st.write("# Model Training")
    st.write("##### 1. VGG16")
    st.write(
"""The VGG16 model is a Convolutional Neural Network architecture that has been widely used for 
image classification tasks.""")
    
    st.write(""" **Observations**:  
             
- The resizing of images for VGG16 to 244,244 RGB dimensions causes alot of the features of the defects to
             be distorted. 
- Unfortunately this model was unable to present reasonably good output results for our segmentation task. 
- Hence we dropped any further training on this pre-trained model""")
    
    st.write("##### 2. RES-UNET")

    st.write("""For the development and implementation of our machine learning model, we went through many 
             design iterations to finally decide on the RES-UNET model scheme.""")
    
    image_13 = load_image('RESUNET_architecture.png')
    st.image(image_13, caption="RES-UNET model with Segmentation and Classification outputs", 
             use_column_width='auto')

    st.write("##### 3. YOLOv5")

    st.write("""In addition to designing and developing our RES_UNET model for training, we also successfully 
             implemented the YOLOv5 object detection model developed by Ultralytics on the PCB datase """)
    st.write("""This pre-trained model can be utilized for both segmentation and classification, providing us 
             with the opportunity to compare the results of our model with this pretrained and well-established 
             design.""")
    
    st.write("For more details on model architecture please go to:")
    st.write("(https://docs.ultralytics.com/yolov5/tutorials/architecture_description/)")

elif page == pages[4]:
    st.write("# Model Optimization and Evaluation")
    st.write("""The performance of the models was evaluated using accuracy, MeanIoU, precision, recall, 
             and F1-score metrics, ultimately achieving a classification accuracy of 95%""")
    st.write("""The classification report and the confusion matrix on the validation set show that precision 
             and recall for each defect class vary but they are generally stable.""")
    
    image_14 = load_image('RESUNET_classification_report_v240618_1.png')
    st.image(image_14, caption="Classification metrics for classification output", 
             use_column_width='auto')
    
    image_15 = load_image('RESUNET_confusion_matrix_v240618_1.png')
    st.image(image_15, caption="Confusion Matrix for classification output", 
             use_column_width='auto')
    
    st.write("##### RES-NET model Results")
    st.write("""The figure below illustrates that the location of the defects or the pixel matrix is predicted 
             quite precisely as Segmentation output. For classification output, the real and predicted classes 
             are shown along with confidence values for the label prediction.""")

    image_16 = load_image('RESUNET_prediction_examples_b_v240618_1.png')
    st.image(image_16, caption="Validation Results", 
             use_column_width='auto')

    st.write("""Finally we look at the prediction results along side the original images to demonstrate a high 
             degree of accuracy and precision, highlighting the effectiveness and potential of this machine 
             learning model.""")  
    image_17 = load_image('results1_resunet.png')
    st.image(image_17, use_column_width='auto')
    image_18 = load_image('results2_resunet.png')
    st.image(image_18, use_column_width='auto')
    image_19 = load_image('results3_resunet.png')
    st.image(image_19, caption="Final prediction Results for RES-UNET model", 
             use_column_width='auto')
         

elif page == pages[5]:
    #image_faiza = load_image('faiza.png')
    #st.image(image_faiza, use_column_width=200)
    st.write("### Faiza Waheed") 
    st.write("""*I am a Communications and Control Systems Engineer with three years of experience as an 
             Instrumentation and Control Systems Engineer. I furthered my education in Germany, 
             earning a Master's Degree in Communications Electronics from the Technical University of Munich. 
             My career then led me to roles as a Testing Engineer at Intel and a Chip Development Engineer 
             at Infineon.*""")
    st.write("""*I am now eager to transition into a dynamic and competitive work environment in Data Science. 
             This career shift represents a new chapter in my unconventional journey, and I look forward 
             to embracing new opportunities and overcoming the challenges ahead.*""")

    #image_niels = load_image('niels.png')
    #st.image(image_niels, use_column_width=200)
    st.write("### Niels Hartano") 
    st.write("""*Enter Niels' background and motivation here.*""")
    
    #image_gernot = load_image('gernot.png')
    #st.image(image_gernot, use_column_width=200)
    st.write("### Gernot Gellwitz") 
    st.write("""*Enter Gernot's background and motivation here.*""")
