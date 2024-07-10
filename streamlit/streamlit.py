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
    current_dir = os.getcwd()
    image_path = os.path.join(current_dir, 'streamlit', 'figures', imageName) # for Streamlit->Github
    #image_path = os.path.join(current_dir, 'figures', imageName)
    image = Image.open(image_path)
    return image

# current_dir = os.getcwd()
# current_dir is not needed outside the functions that define it separately inside themselves

parent_dir = os.path.abspath(os.getcwd()) # for Streamlit->Github
#parent_dir = os.path.abspath(os.pardir)
# I dont understand this. This is not the parent directory but the same as the current dir. And it does not find the image data, if parent_dir is defined this way,
# I therefore have to change parent_dir back, otherwise the app is not running for me.
image_path = os.path.abspath(os.path.join(parent_dir, 'data', 'Images_bb'))


#st.set_page_config(layout="wide", page_title="My Streamlit App")

st.sidebar.title("Table of contents")
pages = ["Project Introduction", "Data Exploration", "Feature Engineering", "Model Training", 
         "Model Optimization and Evaluation", 
         "Authors"]
page = st.sidebar.radio("Go to", pages)

#st.sidebar.header(pages[5])
st.sidebar.markdown("[*Faiza Waheed*](https://github.com/wfaiza/)")
st.sidebar.markdown("[*Niels Hartanto*](https://github.com/taubenus/)")
st.sidebar.markdown("[*Gernot Gellwitz*](https://github.com/Kathartikon/)")

if 'line_index' not in st.session_state:
    st.session_state.line_index = 0

# Function to reveal the next line
def reveal_next_line(text_lines):
    if st.session_state.line_index < len(text_lines):
        st.session_state.line_index += 1

def hide_last_line():
    if st.session_state.line_index > 0:
        st.session_state.line_index -= 1

def local_css(file_name):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    # deleted 'streamlit' since current_dir is already the 'streamlit' folder
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if page == pages[0]:
    st.write('# Detection and Classification of Defects on Printed Circuit Boards (PCBs)')
    #st.html("<h1 style='text-align: center'>Detection and Classification of Defects on Printed Circuit Boards (PCBs) </hr> with Machine Learning</h1>")
    local_css("expander_bold.css")
    with st.expander("Introduction", expanded=False):
        st.write("- This project explores various machine learning methodologies for detecting and classifying defects on PCBs, using advanced computer vision techniques")
        st.write("- PCBs are essential components in nearly all electronic devices.")
        st.write("- Ensuring their quality is critical, as defects can lead to device malfunctions or failures.")
        st.write("- Traditional manual inspection methods are time-consuming and error-prone")
        st.write("- This motivates the adoption of deep learning models such as VGG16, RES-UNET, and YOLOv5 for automated defect detection.")
        st.write("- Our main object was therefore to learn the architecture designing, training and deployment of a manually designed RES_UNET model.")

    image_1 = load_image('PCB-Final-Image.jpg')
    st.image(image_1, caption="Typical 2 layer PCB", use_column_width='auto')

    st.write("### Project Phases")
    #st.write("""The project encompassed several key stages of a rigorous data science methodology. We will cover these aspects of our Data Science project in detail, including:""")
    with st.expander("1 - Data Exploration", expanded=False):
        st.write("- Explore the dataset to understand its structure, features, and potential pitfalls.")
        st.write("- Use data visualization to identify key insights and relevance.")
        st.write("- Ensure the quality of the images.")
       
    with st.expander("2 - Feature Engineering"):
        st.write("- Perform feature engineering to balance and augment the data.")
        st.write("- Create the masks (segmentation) and the target labels (classification) for training images.")
        st.write("- Ensure the target labels were one-hot-encoded as per requirement of classification.")
        st.write("- Implement a randomization strategy to neutralize the impact of baises and ensure model impartiality.")
        st.write("- Ensuring the dataset is ready for model training.")
    
    with st.expander("3 - Model Training"):
        st.write("- Develop various machine learning models to detect the anomalies (defects).")
    
    with st.expander("4 - Model Optimization and Evaluation"):
        st.write("- Refine our designed model to maximize accuracy and robustness.")
        st.write("- Rigorous evaluation of model performance.")
      
    st.write("""This PCB defect detection project has been a dedicated effort for us, blending rigorous data 
analysis and advanced feature engineering with the practical application of machine learning. """)
    st.write("""Our goal was to excel in our endeavor by utilizing all available tools to successfully identify 
and classify PCB defects.""")

    
elif page == pages[1]:
    st.write("# Data Exploration")
    st.write("- The image dataset we used has over 10,000 synthetically generated images.")
    st.write("- The dataset is publicly available at: https://www.kaggle.com/datasets/akhatova/pcb-defects)")
    
    st.write("#### Sample image for a defective PCB:")
    image_2 = load_image('sample_pcb_open_circuit.jpg')
    st.image(image_2, caption="Sample of a defected PCB", width=500)

    with st.expander("The types of defects we will detect in this project:", expanded=False):
        image_3 = load_image('Defect_types.png')
        st.image(image_3, caption="Sample defects explored in this project", use_column_width='auto')

    options = {'Missing Hole':'missing_hole', 'Mouse Bite':'mouse_bite', 'Open Circuit':'open_circuit', 'Short':'short', 'Spur':'spur_', 'Spurious Copper':'spurious_copper'}
    st.markdown('### Sample images with defects')
    choice = st.selectbox('Select Defect', options.keys(), index=0, label_visibility='collapsed')
    if choice is not None:
        img_pool_choice = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if options[choice] in filename]
        rnd_2 = np.random.choice(range(6), 2, replace=False)
        fig = plt.figure(figsize=(15, 30))
        for i, j in enumerate(rnd_2):
            img = cv2.imread(img_pool_choice[j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 2, i + 1)
            plt.axis('off')
            plt.title(f"{choice} {i + 1}", fontsize=20)
            plt.imshow(img)
        st.pyplot(fig)
    st.write("We see that each image can have more than one defect, although in each image we only have one type of defect.")
    local_css('expander_regular.css')
    with st.expander(f"Dimensions of dataset images: {image_2.size}"):
        st.write("⭢ Those images are too large to handle without any pre-processing.")
		
		
    st.write("### Is there a possibilty to minimize the features?")
    #st.write("""Let's inspect the images from the dataset: \n""")

    #image_4 = load_image('Annotated_defects.png')
    #st.image(image_4, caption="Sample image from the PCB dataset with annotated defects", 
    #         width=400)
    
    # sample checkbox for superimposed images
    with st.expander('View superimposed dataset images', expanded=False):
        image_4a = load_image("superimposed_image.png")
        st.image(image_4a, caption='All the training images superimposed together', width=500)
        st.write("- After inspecting the different images, there doesnt seem to be a possibility of reducing features, as this will result in essential features being lost.")
        st.write("- All the various nuances of the image need to be preserved for an accurate and robust anomaly detection model.")

elif page == pages[2]:
    st.write("# Feature Engineering")
    local_css('expander_medium.css')
    st.write("### 1. Visualization of the dataset")
    with st.expander("How balanced is our dataset?"):
        image_5 = load_image('data_balance.png')
        st.image(image_5, caption="Defect distribution", use_column_width='auto')
        st.write(" **Observations**")  
        st.write("- The dataset is relatively balanced overall.")
        st.write("- Still, this is the visualization **before** doing feature engineering to ensure model robustness.")
        st.write("- We need to reevaluate the label distribution again after we performed the next steps.")

    st.write("### 2. Data Augmentation")
    with st.expander("Considerations"):
        st.write("- Data preprocessing plays a crucial role in constructing effective Machine Learning models") 
        st.write("- The quality of prediction results is closely tied to the thoroughness of data preprocessing")
        st.write("- Our image preprocessing pipeline involved several key steps:")
    with st.expander("Dimension Handling"):
        st.write("- Reducing the image dimensions initially from RGB to Grayscale")
        image_6 = load_image('rgb vs grayscale.png')
        st.image(image_6, caption="Colored vs. Grayscale image", use_column_width='auto')
        
        st.write("- Cropping the the image to 100 x 100 grayscale images")
        st.write("- Some defects can be cut into two parts during that process")
        st.write("- That would influence the model training")
        image_6_1 = load_image('image_cropping.png')
        st.image(image_6_1, caption='Image Cropping', use_column_width='auto')


    with st.expander("Mask and target label"):
        st.write("The reference mask for the defects was generated with the help of the detailed annotations provided along with the image dataset. Each bounding box coordinates were mapped onto the mask image and stored in memory along with the reference training image.\nBesides this, the reference target labels were also generated for classification and one-hot encoded.")

        image_6_2 = load_image('defect_img_vs_pm.png')
        st.image(image_6_2, caption="Original Image vs. Pixel Mask", use_column_width='auto')

    with st.expander("Augmentation"):
        st.write("We implemented data augmentation with the help of Albumentations library. This unfortunatly did not cater for the the defects in the optimum way as some defects on the border would be cropped out. Nor did it cater for the instances where multiple defects were located in one image. Hence we had to implement manual augmentation which included:")
    
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
dataset selection so that we could avoid training baises.""")

    st.write(
"""Once we managed to implement the above mentioned feature engineering aspects, we could 
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
    st.image(image_14, caption="Metrics for classification output", 
             use_column_width='auto')
    
    image_15 = load_image('RESUNET_confusion_matrix_v240618_1.png')
    st.image(image_15, caption="Confusion Matrix for classification output", 
             width=500)
    
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
    st.write("""*I am a Communications Engineer with three years of experience as an 
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
