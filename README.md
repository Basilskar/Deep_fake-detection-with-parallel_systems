# Deepfake Detection with Parallel Training  

## Overview  
This project aims to detect deepfake images using a Convolutional Neural Network (CNN) and High-Performance Computing (HPC) techniques. The model is trained using both sequential and distributed training strategies to optimize performance. The dataset is processed through a pipeline that includes compression, error level analysis, and feature extraction before feeding into the CNN for classification.  

## Architecture  

1. **Input Image**: The model takes an image (or video frame) as input.  
2. **Image Compression**: The image undergoes compression to reveal tampering artifacts.  
3. **Difference Calculation**: A difference image is generated using Error Level Analysis (ELA).  
4. **Feature Extraction**: Key features are extracted for deepfake detection.  
5. **Grayscale Conversion & Bit Computation**: The image is converted to grayscale, and pixel values are refined.  
6. **Pixel Value Extraction**: Essential pixel values are extracted for further analysis.  
7. **Reshape Input Image**: The image is reshaped for CNN compatibility.  
8. **Dataset Creation**: Processed images are stored in a dataset for training.  
9. **CNN Model**: A deep learning model classifies images as real or fake.  
10. **Parallel Training**: The model is trained using both sequential and distributed training strategies.  

## CNN Model Architecture  

| Layer Type              | Output Shape         | Parameters |
|-------------------------|---------------------|------------|
| Conv2D (32 filters, 3x3)  | (126, 126, 32)     | 896        |
| BatchNormalization      | (126, 126, 32)      | 128        |
| MaxPooling2D (2x2)      | (63, 63, 32)        | 0          |
| Conv2D (64 filters, 3x3)  | (61, 61, 64)       | 18,496     |
| BatchNormalization      | (61, 61, 64)        | 256        |
| MaxPooling2D (2x2)      | (30, 30, 64)        | 0          |
| Flatten                 | (57600)             | 0          |
| Dense (128 neurons)     | (128)               | 7,372,928  |
| Dropout (0.5)           | (128)               | 0          |
| Dense (2 neurons)       | (2)                 | 258        |

**Total Parameters:** 7,392,962  

## High-Performance Computing (HPC) Setup  

- **Data-Level Parallelism**:  
  - The dataset is split across multiple GPUs.  
  - Synchronous and asynchronous training approaches are compared.  
  - Gradient aggregation is used to update the global model.  

- **Model-Level Parallelism**:  
  - Different layers of the CNN are assigned to different GPUs.  
  - Tensor parallelism and pipeline parallelism techniques are used.  

## Experimental Configuration  

- **Dataset**: CASIA1 (real images) & CASIA2 (tampered images)  
- **Image Size**: 128x128 pixels  
- **Training Parameters**:  
  - Epochs: 30 (early stopping enabled)  
  - Batch Size: 32  
  - Learning Rate: 0.0001 (decay: 0.000001)  
  - Optimizer: Adam  
  - Loss Function: Categorical Crossentropy  
  - Data Split: 80% training, 20% validation  

## Results and Analysis  

### Training Time Comparison  

| Training Method     | Time Taken (seconds) |
|---------------------|---------------------|
| Sequential (5 epochs) | 2436.73            |
| Parallel (30 epochs)  | 196.47             |

### Accuracy Comparison  

| Training Method  | Accuracy |
|------------------|---------|
| HPC Training    | 92.7%   |
| Non-HPC Training | 87.0%   |

### Impact of Error Level Analysis (ELA)  

- Without ELA: Accuracy = **78%**  
- With ELA: Accuracy = **92.7%**  

## Conclusion  
This project demonstrates the effectiveness of deepfake detection using CNNs and HPC techniques. By leveraging distributed training and parallel computing strategies, the model achieves significant improvements in training speed while maintaining high accuracy.
