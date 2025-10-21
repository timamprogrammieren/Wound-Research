# Wound Type Classification

This project applies deep learning to classify wound images.  
A self-collected dataset (~150 images) is split into train/test sets.  
The model combines EfficientNet-B3 and MobileNetV3-Small with attention modules (ECA, scSE) for better accuracy and interpretability.  
Code is written in PyTorch and includes training, evaluation, and Grad-CAM visualization.
