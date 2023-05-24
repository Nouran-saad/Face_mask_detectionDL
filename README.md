# Face_mask_detectionDL

<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/47dc65ff-1817-4c3b-9f3c-db4871e3f806"  width="30%" height="5%" align="right">

***CSE485 Deep Learning***
## Problem Statment
Implementing a face mask detection system requires integrating multiple technologies, including computer vision, machine learning, and deep learning. The goal is to create an automated system that can analyse video frames, identify faces, and accurately classify whether a person is wearing a mask, wearing it incorrectly, or not wearing a mask at all. The system should provide real-time feedback to ensure timely intervention and enforcement of mask-wearing protocols.
To address this problem, we will utilize Python, Keras, and OpenCV, which are popular tools and libraries for computer vision and deep learning tasks. By leveraging pre-trained deep learning models and image processing techniques, we can develop an efficient and effective face mask detection system that operates on real video streams.

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
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/f98b4811-5a38-40a0-866c-6e445bd2877e"  width="50%" height="5%" align="right">
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

* Classification Report
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/7820b251-d288-418c-8332-aa8095a84dc4"  width="50%" height="5%" align="right">
</p>
<br />
<br />
<br />
<br />
<br />
<br />

* Graph of loss and accuracy
<p align="center">
<img src="https://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/e5f2b45c-1c2b-40e1-a8c7-9db87cf80813"  width="50%" height="5%" align="right">
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
<img src="ttps://github.com/Nouran-saad/Face_mask_detectionDL/assets/55962261/d98018ed-df28-4f11-bcd7-9328a41e2ca6"  width="50%" height="5%" align="right">
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


## Comparison between other architectures
1. **MobileNetV2**
   * Pros: MobileNetV2 is lightweight and efficient, making it suitable for deployment on resource-constrained devices. It offers a good balance between model size, speed, and accuracy.
   * Cons: Due to its reduced complexity, MobileNetV2 may have limitations in capturing intricate details and handling complex patterns compared to deeper architectures.
2. **ResNet**
   * Pros: ResNet's residual connections enable effective training of very deep networks and help capture intricate features. It has achieved state-of-the-art performance on various computer vision tasks.
   * Cons: Deeper ResNet architectures may be computationally expensive and require a larger amount of training data to prevent overfitting.
3. **DenseNet**
   * Pros: DenseNet's dense connectivity promotes feature reuse and enhances gradient flow throughout the network. It allows for efficient parameter usage and has shown strong performance with limited training data.
   * Cons: DenseNet architectures may have higher memory requirements compared to other models due to the dense connections.
4. **Inception**
    * Pros: Inception models capture information at multiple spatial resolutions, allowing them to extract both local and global features effectively. They have demonstrated strong performance in various computer vision tasks.
    * Cons: Inception architectures can be computationally expensive and may require more resources during training and inference compared to simpler models.

