# License Plate Detection And Recognition Tool

# Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Usage](#usage)
- [Methodology](#methodology)
- [Conclusion](#conclusion)

# Overview
This project covers a License Plate Detection and Recognition Tool built upon YOLOv4/YOLOv7 for license plate detection and PaddleOCR for license plate character recognition. It allows the processing of images and videos to accurately detect license plates and extract their characters, enabling a variety of applications such as traffic monitoring, parking management, and law enforcement.

# Demo
![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/demo_video.gif)
![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/carpark_demo.png)

# Features
- License plate detection using state-of-the-art YOLOv4/YOLOv7 models.
- Accurate license plate character recognition using PaddleOCR.

# Usage
You can access the central notebook for training the YOLOv7 license plate detection model and executing the detector and character recognition (PaddleOCR): [YOLOv7 License Plate Detection & Recognition](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/yolo_v7_license_plate_detection.ipynb)

An earlier iteration that uses YOLOv4 with Darknet and PaddleOCR: [YOLOv4 License Plate Detection & Recognition](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/ALPR_Implementation.ipynb)

# Methodology
- [Objective](#objective)
- [Data Collection](#data-collection)
- [Preprocessing & Annotation](#preprocessing--annotation)
- [License Plate Detection](#license-plate-detection)
- [License Plate Recognition](#license-plate-recognition)

## Objective
License plate detection and character recognition of images and videos. A sequential methodology was employed, consisting of a two-step process: initially utilizing the one-stage detector YOLO for license plate detection, followed by subsequent Optical Character Recognition (OCR) performed using PaddleOCR on the identified license plates.

## Data Collection
- [Google Open Images Dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F01jfm_) (Baseline Data)

The Google Open Images Dataset encompasses a pre-annotated collection comprising 1500 training images and 300 validation images, meticulously organized in the YOLO format. These images capture a variety of vehicles with their respective license plates from various global locations. However, it's important to note that the dataset does not include instances of double-line license plates, a prevalent type of license plate found in Singapore.

- [nuScenes](https://www.nuscenes.org/nuimages#download) (Singapore Data)
- [Street Driving Videos](https://www.youtube.com/watch?v=vBySF9eSKQs&ab_channel=JUtah) (Singapore Data)

To address the constraints of the Google Open Images Dataset and facilitate enhanced training on Singapore-specific data, we can leverage the nuScenes and Street Driving Videos datasets. By integrating these additional datasets, our model gains the capacity to capture nuances and intricacies unique to Singapore's license plate characteristics.

- Carpark CCTV Footage (CCTV Data)

The Carpark CCTV Footage Dataset plays a pivotal role in training the model to recognize license plates under distinct conditions. By exposing the model to the distinctive angles and image qualities prevalent in CCTV camera footage, we ensure its proficiency in identifying license plates accurately even in challenging scenarios.

To reduce the time taken for data collection, videos can be split into frames with the file `vid2img.py` located [here](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/License-Plate-Recognition-YOLOv7-and-CNN-main/vid2img.py).

The datasets were split into 70/20/10 for the train-validation-test split.

## Preprocessing & Annotation
Image preprocessing is not required as YOLO automatically resizes the images. Annotations for images were done with [labelImg](https://github.com/HumanSignal/labelImg#create-pre-defined-classes) in the YOLO format where we will have a `.txt` file in the same folder with the image. The `.txt` file contains the labels for the normalized bounding box coordinates(x,y,w,h) of all license plates found in the image.

![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/labelImg_demo.png)

## License Plate Detection
### YOLOv4
YOLOv4 was implemented with the Darknet framework, an open-source neural network framework written in C and CUDA. YOLOv4 uses CSPDarknet53 CNN which means, its backbone for object detection uses Darknet53 which has a total of 53 convolutional layers. 

Achieved Mean Average Precision (mAP) of 90% when trained on Google's Open Image Dataset. A notable limitation for this iteration is the model's inability to detect double-line plates.

### YOLOv7
With better real-time license plate detection, YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100.

Attained a Mean Average Precision (mAP) of 90% through a systematic transfer learning approach. This was realized by executing multiple training iterations, commencing with a foundational baseline training using Google's Open Image Dataset, and subsequently progressing to Singapore-centric datasets to fine-tune the model.

![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/train_demo.png)
![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/train_demo2.png)

#### Training Configurations & Considerations
**1. Batch Size**

When configuring your training, the choice of batch size plays a pivotal role in optimizing the learning process. Different batch sizes offer distinct advantages and considerations:

- **Larger batch size:** Leveraging a larger batch size can harness the parallel processing capabilities of GPUs, potentially leading to expedited training times. However, be cautious with very large batch sizes, as they might demand substantial GPU memory, impacting the maximum feasible batch size.

- **Smaller batch size:** Opting for smaller batch sizes proves beneficial when GPU memory is constrained, as they consume less memory. Yet, excessively diminutive batch sizes can yield noisy gradients and protracted convergence.

In practice, adhere to this rule of thumb: initiate training with a moderately sized batch that fits comfortably within your GPU memory. Subsequently, experiment with both larger and smaller batch sizes to gauge their effects on training speed and convergence. Frequently observed batch sizes are often powers of 2, such as 8, 16, 32, and 64. In my iterations I used a batch size of 16.

**2. Number of Epochs**

An epoch refers to one complete pass through the entire training dataset. During each epoch, the model will be fed with different batches of data for optimization, with the aim to improve its ability to detect objects accurately.

- **Too few epochs:** If you train for too few epochs, the model may not have enough time to converge and reach its optimal performance. The model might underfit and not generalize well to new data.

- **Too many epochs:** Training for too many epochs may lead to overfitting, where the model becomes too specialized to the training data and performs poorly on new, unseen data.

Monitor your model's performance on a validation set during training. Typically, you should stop training when the validation loss stops decreasing or starts to increase (indicating overfitting). You can also use techniques like learning rate scheduling and early stopping to help with finding the optimal number of epochs.

#### Training Metrics
**1. mAP (mean Average Precision)**

Evaluation metric used to assess the performance of object detection models. It quantifies how well the model detects objects of interest, and it takes into account both precision and recall.

**mAP@.5**:
- This metric calculates the mean Average Precision when the IoU (Intersection over Union) threshold is set to 0.5.
- IoU is a measure of how well the predicted bounding boxes overlap with the ground-truth bounding boxes.
- When mAP is calculated with an IoU threshold of 0.5, it means that a predicted bounding box is considered correct if it has an IoU of 0.5 or higher with the corresponding ground-truth bounding box.
- A higher mAP@.5 score indicates that the model is good at localizing objects, and at least 50% of the object's region is correctly predicted.

**mAP@.5:.95**:
- This metric calculates the mean Average Precision over a range of IoU thresholds from 0.5 to 0.95.
- Instead of considering a single IoU threshold (as in mAP@.5), this metric provides a more comprehensive evaluation across a range of IoU values.
- The IoU threshold is gradually increased from 0.5 to 0.95, and the precision and recall are computed for each threshold.
- The final mAP@.5:.95 score is the average of all the calculated Average Precisions across the IoU range.
- A higher mAP@.5:.95 score indicates that the model performs well across various levels of strictness in bounding box matching.

In summary, both mAP@.5 and mAP@.5:.95 are used to evaluate the object detection model's performance, but mAP@.5:.95 provides a more detailed assessment by considering multiple IoU thresholds, reflecting how well the model performs at different levels of bounding box overlap. Higher mAP scores generally indicate better object detection performance.

**2. Best Possible Recall (BPR)**

A metric that measures the proportion of actual positive samples (objects) that are correctly identified by the model. "Best Possible Recall (BPR) = 1.0000" means that the model achieved a recall of 1.0000, which indicates that it correctly detected all the objects in the dataset.

**3. box, obj, cls, total**

These columns represent different loss values during training. Losses are measures of how well the model is performing during each epoch.

- **box:** Represents the localization loss, which measures the accuracy of predicting the bounding box coordinates.
- **obj:** Stands for the objectness loss, measuring the confidence of predicting whether an object is present in the bounding box.
- **cls:** Denotes the classification loss, which measures the accuracy of predicting the class label of the object within the bounding box.
- **total:** The total loss, which is the sum of all the individual losses (box, obj, cls).

**4. Class Metrics**

- **Class:** The name of the object class being evaluated.
- **Images:** The number of images in the dataset containing instances of the specific class.
- **Labels:** The total number of instances (bounding boxes) of the specific class present in the dataset.
- **P:** Precision, which measures the percentage of correctly predicted positive instances out of all predicted instances of the class.
- **R:** Recall, which measures the percentage of correctly predicted positive instances out of all instances of the class present in the dataset.
- **"all"**: Refers to the aggregated metrics for all object classes, not just for a specific class.

## License Plate Recognition
Instead of employing the Hough Transform alignment along with character segmentation and recognition to extract characters from the detected license plate, an alternative approach was adopted. Optical character recognition (OCR) using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) was chosen as the method to accurately decipher characters from the identified license plate.

PaddleOCR uses the CRNN recognition algorithm. CRNN (Convolutional Recurrent Neural Network) is a popular architecture for text recognition tasks, including optical character recognition (OCR). It combines convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to effectively handle the sequential nature of text while capturing spatial information from input images.

![](https://github.com/xavierkoo/computer_vision_anpr_alpr/blob/main/content/two_line_demo.png)
