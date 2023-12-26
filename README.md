Project Title
Event Detection in Videos using CNN-LSTM Hybrid Model
Introduction
Motivation
Thanks to advancements in camera and smartphone recording capabilities, video recording has grown in popularity in recent years. Along with the support from platforms like YouTube and TikTok, this evolution has made it easier and more accessible to shoot and share high-quality video content for both professional and independent creators. As a result, there is an increasing amount of video being produced, leading to a growing demand for tools to effectively organize, categorize, and analyze this content.

Objective
This project aims to explore how we handle video data for archival and creative purposes, through automatic event detection. Utilizing a CNN-LSTM hybrid model, which combines Convolutional Neural Networks with Long Short-Term Memory, this project aims to accurately detect events in video by training on the VidLife dataset featuring “The Big Bang Theory”.

Dataset
The dataset used in this project is VidLife - a video life event extraction dataset that uses footage from the popular American TV series ‘The Big Bang Theory’. VidLife contains 14,343 training examples, 1,793 examples for validation, and 1,793 examples for testing. However, only part of the dataset is available for use, amounting to 4,198 examples. The VidLife dataset offers a comprehensive collection of video clips extracted into frames at a rate of 3 frames per second (FPS), capturing a total of 43 different events with 29 sorts of high-level events and 14 sorts of low-level events.

Methodology
CNN-LSTM Model
The model that we are building in this project is a hybrid model combining Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks, in short, CNN-LSTM. This combination is most suitable to capture both spatial and temporal features of the data. The model also integrates Max-Pooling layers to reduce spatial dimensions, a flattening step to convert the data into a suitable format for the LSTM, a fully connected layer, and a sigmoid activation function for the final classification. The comprehensive overview of the model architecture is illustrated in the project.

Training
The model was trained for 5 epochs, taking approximately 40 hours. Training and validation loss of the neural network is computed using the Binary Cross-Entropy (BCE) Loss function. The Adam optimizer, with a learning rate of 0.001, utilizes these gradients to update the weights in an attempt to minimize the loss in subsequent iterations.

Results
To assess the model’s performance, precision, recall, F-1, and accuracy scores are evaluated. The results are considered in two sets:

Overall results of 43 classes: Precision, recall, and F-1 score are significantly lower, while accuracy remains high. This indicates that the model is mostly predicting negative cases, and fails to recognize positive ones. This can be due to an imbalance in the dataset, where some classes are underrepresented, making it harder for the model to learn.
Results of the top 10 classes with sufficient data: Precision, recall, and F-1 score are higher, while accuracy is moderately low. This indicates that the model can identify positive cases more frequently but creates more false positives.
Conclusion
The CNN-LSTM hybrid model shows promising potential in identifying and classifying events in videos, with fairly positive results in well-represented classes. Data scarcity remains a significant challenge, as underrepresentation of many classes in the dataset leads to weaker overall performance. Overall, this project presents an opportunity for further research and development when the full VidLife dataset becomes available.
