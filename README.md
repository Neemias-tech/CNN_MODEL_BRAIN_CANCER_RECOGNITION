# CNN_MODEL_BRAIN_CANCER_RECOGNITION

📌 Project Overview
This project leverages deep learning to classify MRI brain scans into tumor (yes) or no tumor (no) categories. The goal is to develop an automated and accurate model to assist in brain tumor detection using computer vision techniques.

A Convolutional Neural Network (CNN) was trained on a dataset of MRI scans and evaluated based on accuracy, precision, recall, and F1-score. The model was then used to predict and visualize results on test images.

🎯 Project Goals
✅ Build an image classification model using deep learning
✅ Train a CNN to classify MRI scans as tumor or no tumor
✅ Evaluate the model’s performance with key metrics
✅ Visualize actual vs. predicted classifications for test images
✅ Improve model accuracy through fine-tuning and augmentation
🛠️ Model Architecture
The Convolutional Neural Network (CNN) consists of:

Convolutional Layers: Extract features from MRI scans
Pooling Layers: Reduce dimensionality while preserving information
Dropout Layers: Prevent overfitting
Dense Layers: Classify images into yes or no
🔹 Optimizer: Adam
🔹 Loss Function: Binary Cross-Entropy
🔹 Activation Functions: ReLU & Sigmoid
🔹 Evaluation Metrics: Accuracy, Precision, Recall, F1-score
📌 Key Achievements
✅ Successfully trained a CNN to classify MRI scans
✅ Achieved 77% accuracy on the test dataset
✅ Visualized predictions using matplotlib
✅ Saved and loaded the trained model for inference
✅ Developed a scalable deep learning pipeline

🚀 Future Improvements
🔹 Fine-tune the CNN model with more layers
🔹 Experiment with data augmentation to boost performance
🔹 Implement transfer learning with pre-trained models (VGG16, ResNet)
🔹 Deploy the model as a web app for real-time predictions

📜 License
This project is licensed under the MIT License. Feel free to use and modify it for research and educational purposes.
