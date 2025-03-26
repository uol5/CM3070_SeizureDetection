edf_to_numpy.ipynb -
Purpose: Early version for extracting EEG data from EDF files into NumPy format.

Features:

Uses mne to load EEG data.

Splits data folders into training and testing sets.

Extracts 10-second windows across all recordings.

Seizure Proportion Handling:

Captures around 27% seizure data.

No explicit class balancing or seizure upsampling logic.

Bandpass Filtering:

Not implemented in this version. Preprocessing is minimal.

----------------------------------------------------------------------------------------------------------------------

EdfToNumpy2.ipynb -
Purpose: Updated version with improved preprocessing and seizure balancing.

Features:

Organized using a class (EdfToNpy) with clear constants and methods.

Applies bandpass filtering (1–20 Hz) using scipy.signal.butter and filtfilt.

Uses 10-second overlapping windows.

Seizure Proportion Handling:

Achieves 43% seizure data.

Implements controlled sampling via a SEIZURE_PROPORTION parameter to balance classes by reducing non-seizure data.

Bandpass Filtering:

Applied here as a preprocessing step; later moved to main.py during model development.

--------------------------------------------------------------------------------------------------------------------

seizuretools.py -

EEG Preprocessing & Seizure Detection Utilities
This module provides a comprehensive toolkit for processing scalp EEG data and building deep learning models for seizure detection. Tailored for use with the CHB-MIT dataset or similar EEG recordings, it includes:

Preprocessing Tools:

Bandpass filtering (1–20 Hz) to isolate relevant brainwave frequencies

Z-score normalization for signal standardization

EDF file parsing and window-based segmentation with seizure proportion control

Model Architecture:

A hybrid ConvBLSTM (Convolution + Bidirectional LSTM) deep learning model designed for binary seizure classification

Utilities:

Functions for data preparation, training, evaluation, and visualization

Pretrained model loading and real-time EEG segment prediction with optional plotting

The module is structured for both research and clinical prototyping, enabling end-to-end EEG model workflows.

-------------------------------------------------------------------------------------------------------------------

main.py –

Model Training & Seizure Prediction Pipeline
This script orchestrates the full machine learning workflow for EEG-based seizure detection. It includes:

Training of supervised, unsupervised (autoencoder + clustering), and recurrent neural network (CNN-BiLSTM) models

EEG preprocessing using bandpass filtering and normalization

Evaluation using metrics like recall, precision, accuracy, and confusion matrix

Model saving and deployment-ready prediction interface for new EEG data

main.py serves as the core engine for developing and validating seizure detection models using preprocessed data from seizuretools.py.

---------------------------------------------------------------------------------------------------------------