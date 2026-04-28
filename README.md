# SEMG-based Hand Gesture Recognition using Hybrid Deep Learning
This repository contains the implementation of a hybrid deep learning framework designed to decode surface Electromyography (SEMG) signals for high-accuracy hand gesture recognition. The project focuses on addressing inter-user variability through a specialized transfer learning and calibration strategy.
## 🌟 Key Features
 * *Hybrid Architecture:* Integrates 1D-CNN (spatial features), BiLSTM (temporal sequences), and Multi-head Self-Attention (contextual importance).
 * *User-Adaptive Calibration:* Employs a pre-training and fine-tuning strategy, achieving high accuracy on unseen subjects with minimal data (only 20% for calibration).
 * *Robust Preprocessing:* Includes digital Butterworth bandpass filtering and time-domain feature engineering (RMS, MAV, SSI, VAR).
## 🏗️ Model Architecture
The model follows a hierarchical structure to process 8-channel EMG inputs:
 1. *Spatial Feature Extraction (1D-CNN):* A multi-layer CNN (64, 128 filters) captures hierarchical spatial patterns from the sensor array.
 2. *Temporal Sequence Modeling (BiLSTM):* A bidirectional LSTM layer processes data in both forward and backward directions to understand the progression of muscle activity.
 3. *Contextual Weighting (Self-Attention):* A 4-head self-attention mechanism automatically focuses on the most informative segments of the signal window.
 4. *Classification:* Dense layers with Dropout (0.5) and Softmax activation map the learned features to specific gesture classes.
## 🛠️ Project Pipeline
 1. *Signal Denoising:* 4th-order Butterworth bandpass filtering (10Hz - 99Hz) to remove motion artifacts and power line interference.
 2. *Segmentation:* Sliding window technique with a size of 200 samples and a high-overlap step size.
 3. *Feature Engineering:* Calculation of four time-domain features (Root Mean Square, Mean Absolute Value, Simple Square Integral, and Variance) per channel.
 4. *Training Strategy:*
   * *Phase 1 (Pre-training):* General knowledge acquisition using data from a population group (Users 1-9).
   * *Phase 2 (Fine-tuning):* Personalization for the target user using a low learning rate (1e-5).
## 📊 Experimental Results
 * *Dataset:* RSE-emg-data (10 users, multiple gestures including Closed Fist, Index Extension, etc.).
 * *Performance:* Achieved *88.89% Accuracy* on an unseen user (User 10) after the calibration phase.
 * *Reliability:* Demonstrated stable convergence and effective discrimination between similar muscle firing patterns.
## 🚀 Getting Started
### Prerequisites
 * Python 3.8+
 * TensorFlow / Keras
 * Scikit-learn, Pandas, Numpy, Scipy
### Usage
 1. Clone the repository:
 ```bash
   git clone https://github.com/yourusername/emg-gesture-recognition.git
```
 2. Install dependencies:
```bash
   pip install -r requirements.txt
```   
   
 3. Run the analysis:
   * Open RSE-Data.ipynb in Google Colab or Jupyter Notebook.
   * Configure the path variable to point to your local dataset directory.
## 👥 Contributors
 * *Tran Viet Bach* - Hanoi University of Science and Technology (HUST).
 * *Bui Phuong Duy* - Hanoi University of Science and Technology (HUST).