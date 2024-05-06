# PCB_Defects_Detection

This project undertakes the tasks of detecting defects on PCB (Printed Circuit Boards) from a large database of images.
Dataset location: https://robotics.pkusz.edu.cn/resources/datasetENG/

PCB Defect Detection Using Image Processing and CNN deep learning framework

Introduction: 
Visual inspection, defect detection and recall are some of the most complex and expensive tasks for PCB manufacturing companies. Over the years, Printed Circuit Boards have become much smaller and more densely packed with components making the scalability of visual inspection harder. With increased demands from the electronics industry, many defects go unnoticed, which may lead to poor repute and reduced business.  

Project resource:
In this project, I propose to use deep learning with visual object detection to automatically detect 4-6 defects on a Printed Circuit Board. PCB Dataset details are elaborated below: 
A public PCB dataset containing 1368 images with 6 kinds of defects (Missing hole; Mouse bite; Open circuit; Short circuit; Spurious copper; Spur) will be used for detection, classification and reporting tasks. This dataset is provided for public use and hosted on Kaggle.com, which is a community for data scientists and ML developers.
Dataset is located at https://www.kaggle.com/datasets/akhatova/pcb-defects
The dataset hosted on Kaggle is affectively sourced from the Open Lab on Human Robot Interaction of Peking University. https://robotics.pkusz.edu.cn/resources/datasetENG/
 
Steps involved in the project: 

Obtaining and verifying the Dataset 

Model training on regression model 

Testing and validating results

Model training CNN deep learning model 

Testing and validating results

Optimizing deep learning model for performance 

Configuring project with Streamlit 


========================================================================================================================================


Project description + Visualizations for 1st Feedback meeting:

To fabricate the PCB; Cutting, drilling, placing graphics, etching, putting on a solder mask, and printing are all part of the process.
A PCB (Printed Circuit Board) consists of the following components.
Diagram of PCB with image description:

![componentside-pcb-circuit](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/c822ca4d-355b-4b9c-87a2-e002b11decb6)

Figure 1 : PCB with Electronic Components

PCB's may be designed as single-layer, double-layer or multi-layer. In this preject we will be dealing with double-layer design with the images in 2 dimension only.
Double layer PCB comes with 2 layers of conducting material on both sides of the board. Each side is also used for incorporating different electronic components on top of the board. 
Double layer PCB's are widely used in a variety of different electronic sectors where low costs are required.


![layer 1](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/37212d01-69d2-40f4-92b2-72090e94370d)

Figure 2 : Double layer PCB with Top layer view

![layer 2](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/31368ae5-c8c2-4933-a6a7-e661a5452a16)

Figure 3 : Double layer PCB with Bottom layer view

![Designed-Printed-Circuit-Board-PCB-bottom-layer-top-layer](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/ebbcecaa-3f87-4b26-92e7-f910058ef4d2)

Figure 4 : Designed Printed Circuit Board PCB bottom layer top layer

![PCB-Final-Image](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/b0990ba5-1592-44d1-9120-0abf5993361d)

Figure 5 : PCB without components attached


In a single fabricated PCB, there can be quite a few types of defects. Out of the multitudes of defects, we will concern ourselves with the following 6 defects:

![PCB_defects_2](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/8672bd90-05d7-4113-8502-489ecdc3677b)

Figure 6 : (a) missing hole; (b) mouse bite; (c) open circuit; (d) short circuit; (e) spur; (f) spurious copper.

Before the components are attched onto the board, the routes/traces are etched. These etches complete the circuit according to the design. 
Most of the defects (mouse bite; open circuit; short circuit; spur; spurious copper) occur at this stage but can only be detected once the PCB is manufactured and ready for component soldering.

![Original vs  Bounded Defects](https://github.com/wfaiza/PCB_Defects_Detection/assets/142170637/8c9b208c-625e-442b-a71d-8a40e04222eb)

Figure 7 : PCB with defects



