# Face_mask_detectionDL

<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/47dc65ff-1817-4c3b-9f3c-db4871e3f806"  width="30%" height="5%" align="right">

***CSE485 Deep Learning***
## Problem Statment
Implementing a face mask detection system requires integrating multiple technologies, including computer vision, machine learning, and deep learning. The goal is to create an automated system that can analyse video frames, identify faces, and accurately classify whether a person is wearing a mask, wearing it incorrectly, or not wearing a mask at all. The system should provide real-time feedback to ensure timely intervention and enforcement of mask-wearing protocols.
To address this problem, we will utilize Python, Keras, and OpenCV, which are popular tools and libraries for computer vision and deep learning tasks. By leveraging pre-trained deep learning models and image processing techniques, we can develop an efficient and effective face mask detection system that operates on real video streams.

## Diagram
![image](https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/192bb94b-6e90-4169-ba6d-7309fa7d3502)

## Implementation
* Architecture of MobileNetV2 (Detection of mask):
The inverted residual structure is introduced as an alternative to the traditional residual connections. Instead of adding the input feature maps to the output, MobileNetV2 employs a depth-wise convolution which is , it applies a separate filter to each input channel and produces a set of intermediate output channels followed by a point-wise linear projection to change the number of channels and It takes the intermediate output channels from the depth-wise convolution and performs a 1 × 1 convolution on them. This convolution is responsible for combining and building new features by computing linear combinations of the input channels. This structure reduces the number of parameters and computations compared to traditional residuals.
MobileNetV2 also introduce the concept of linear bottlenecks, which are used to increase the non-linearity of the network without adding additional parameters. By applying a lightweight non-linearity, such as a ReLU (Rectified Linear Unit), after the depth-wise convolution, the network can capture more complex features while keeping the model size small.

<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/8386b3d9-b972-4305-b914-ad44e747820f"  width="60%" height="5%" align="right">

<br />
<br />
<br />
<br />
<br />
<br />


## Testing
* Result of training throughout the 12 epochs
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/891bdfa9-8fac-46d6-b915-82aaa7f04e5d"  width="50%" height="5%" align="right">
</p>

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />



* Graph of loss and accuracy
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/e71bdf8f-3532-48e8-9960-b403d965fa81"  width="50%" height="5%" align="right">
</p>
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />


* Result of video stream with mask and no mask
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/d19a5271-a24d-4727-a2aa-fdaae6035207"  width="50%" height="5%" align="right">
</p>
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/d98018ed-df28-4f11-bcd7-9328a41e2ca6"  width="50%" height="5%" align="right">
</p>

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

* Result of images with mask and no mask
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/8034fa61-5243-4fa0-92ef-271e824af43e"  width="30%" height="2%" align="right">
</p>

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/85051949-ce54-4511-90bc-2e1551025dbe"  width="30%" height="2%" align="right">
</p>


<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />



## Comparison between other architectures
1.  **MobileNetV2**
We can see that it’s accurate model with accuracy 97.33% and when we tested it on images it worked well with a good prediction value but still resNet50 better than it.
The model takes to finish the 12 epochs 18 minutes and it the shortest time between all the models. 
2.  **ResNet**
We can see that it’s the most accurate model with accuracy 99.08% and when we tested it on images it worked so well with the highest prediction values among the remaining models. 
The model takes to finish the 12 epochs 53 minutes and it the longest time between all the models. 
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/2348f839-ef16-419f-ad51-37bcb4d8cfc3"  width="50%" height="3%" align="right">
</p>

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

3. **InceptionV3**
   We can see that it’s the most accurate model with accuracy 92.75 % and when we tested it on images it worked so well with the lowest prediction values among the other models. 
The model takes to finish the 12 epochs 34 minutes and it is an average time between the other models.
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/f38a832c-3b2b-450c-a997-923919429347"  width="50%" height="3%" align="right">
</p>


