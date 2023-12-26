# Event Detection in Videos using CNN-LSTM Hybrid Model

## Introduction

### Motivation
Thanks to advancements in camera and smartphone recording capabilities, video recording has grown in popularity in recent years. With the support from platforms like YouTube and TikTok, this evolution has made it easier and more accessible to shoot and share high-quality video content for both professional and independent creators. As a result, there is an increasing amount of video being produced, leading to a growing demand for tools to effectively organize, categorize, and analyze this content.

### Objective
This project aims to explore video data handling for archival and creative purposes through automatic event detection. Utilizing a CNN-LSTM hybrid model, which combines Convolutional Neural Networks with Long Short-Term Memory, this project aims to accurately detect events in video by training on the VidLife dataset featuring "The Big Bang Theory".

## Dataset
The dataset used in this project is VidLife - a video life event extraction dataset using footage from the popular American TV series 'The Big Bang Theory'. VidLife contains 14,343 training examples, 1,793 examples for validation, and 1,793 examples for testing. However, only part of the dataset is available for use, amounting to 4,198 examples. The VidLife dataset offers a comprehensive collection of video clips extracted into frames at a rate of 3 frames per second (FPS), capturing a total of 43 different events with 29 sorts of high-level events and 14 sorts of low-level events.

## Methodology

### CNN-LSTM Model
The model in this project is a hybrid combining Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks, termed CNN-LSTM. This combination is most suitable to capture both spatial and temporal features of the data. The model integrates Max-Pooling layers to reduce spatial dimensions, a flattening step for LSTM suitability, a fully connected layer, and a sigmoid activation function for final classification.

### Training
The model was trained for 5 epochs, taking approximately 40 hours. Training and validation loss is computed using the Binary Cross-Entropy (BCE) Loss function. The Adam optimizer, with a learning rate of 0.001, is used to update weights and minimize loss.

## Results
To assess the model’s performance, precision, recall, F-1, and accuracy scores are evaluated in two sets:
- **Overall results of 43 classes**: Lower precision, recall, and F-1 score, with high accuracy. Indicates predominant negative case prediction and failure to recognize positive cases, likely due to dataset imbalance.
- **Results of top 10 classes with sufficient data**: Higher precision, recall, and F-1 score, with moderately low accuracy. Indicates better positive case identification but more false positives.

## Conclusion
The CNN-LSTM hybrid model shows promise in video event identification and classification, especially in well-represented classes. Data scarcity and underrepresentation in the dataset are significant challenges, affecting overall performance. This project highlights the potential for further research and development with the full VidLife dataset.

