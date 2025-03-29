
"""
seizuretools.py

This module provides EEG preprocessing, data extraction, model training, and prediction tools
for seizure detection using deep learning.

Included features:

Preprocessing:
- Bandpass filtering (1–20 Hz) to isolate clinically relevant brainwave frequencies
- Z-score normalisation for amplitude scaling across EEG recordings

Model Architecture:
- ConvBLSTM: A predefined hybrid CNN–Bidirectional LSTM model for spatial-temporal seizure classification

Training and Evaluation:
- Model compilation, training with early stopping, and recall plotting
- Performance evaluation with accuracy, recall, precision, and confusion matrix visualisation

Data Handling (EDF):
- EdfToNpy: Class for processing EDF files to extract seizure and non-seizure EEG segments
- Sliding window segmentation and label assignment using .seizures annotation files
- Data saving as NumPy arrays
- EEG signal visualisation with labelled plot

Prediction:
- SeizurePredictor: Subclass of EdfToNpy to support:
    - Batch or single-sample prediction from raw EEG segments
    - Preprocessing (filtering, normalisation, reshaping) before inference
    - Visual display of predicted seizure segments

Designed for researchers and clinicians working with EEG data (e.g., CHB-MIT dataset)
to streamline seizure detection workflows.
"""


import os
import re
import random
import numpy as np
import mne
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     GlobalAveragePooling2D, Reshape, Bidirectional,
                                     LSTM, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=256.0, order=4):
    """
    Applies a Butterworth bandpass filter to multichannel EEG data.

    Parameters:
        - data (np.ndarray): EEG signal with shape (samples, channels, features)
        - lowcut (float): Lower cutoff frequency in Hz (default: 1.0 Hz)
        - highcut (float): Upper cutoff frequency in Hz (default: 20.0 Hz)
        - fs (float): Sampling frequency in Hz (default: 256 Hz)
        - order (int): Order of the Butterworth filter (default: 4)

    Returns:
        - filtered_data (np.ndarray): Bandpass-filtered EEG data with the same shape as input
    """
    # --- Input validation ---
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError(f"Input array must be 3D (samples, channels, features), got shape {data.shape}.")
    if lowcut >= highcut:
        raise ValueError("Lowcut frequency must be less than highcut frequency.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")

    # Nyquist frequency is maximum frequency reproducible without distortion
    nyquist = 0.5 * fs
    # butter works with frequency normalised between 0-1 of nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Get filter coefficients
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to each channel in the data (data.shape = (samples, channels, features))
    filtered_data = np.zeros_like(data)

    # Apply the filter to each channel across all samples
    # Loop over channels (axis=1), apply filter across time axis (axis=2)
    for i in range(data.shape[1]):
        filtered_data[:, i, :] = filtfilt(b, a, data[:, i, :], axis=1)

    return filtered_data


def normalise_data(x_train, x_test=None):
    """
    Z-score normalisation for EEG input.
    Can operate on train/test together or standalone.

    Args:
        x_train (np.ndarray): Training data or single input to normalize.
        x_test (np.ndarray, optional): Test data to normalize. Defaults to None.

    Returns:
        tuple: (x_train_scaled, x_test_scaled) if x_test is provided,
               else (x_train_scaled, None)
    """
    scaler = StandardScaler()

    # Reshape for scaling: (samples, channels, features) → (samples, -1)
    orig_shape = x_train.shape
    x_train_reshaped = x_train.reshape(orig_shape[0], -1)
    x_train_scaled = scaler.fit_transform(x_train_reshaped).reshape(orig_shape)

    if x_test is not None:
        test_shape = x_test.shape
        x_test_reshaped = x_test.reshape(test_shape[0], -1)
        x_test_scaled = scaler.transform(x_test_reshaped).reshape(test_shape)
        return x_train_scaled, x_test_scaled
    else:
        return x_train_scaled, None


class ConvBLSTM():
    """
    A hybrid Convolutional and Bidirectional LSTM (CNN-BiLSTM) model for seizure classification using EEG data.

    Combines spatial feature extraction (via Conv2D) with temporal sequence modeling (via BiLSTM) for improved
    performance on time-series EEG signals.
    """

    def __init__(self, input_shape=(23, 2560, 1), learning_rate=0.0005, lstm_units=64, dropout_rate=0.3):
        """
        Initializes the ConvBLSTM model with default parameters.

        Parameters:
            input_shape (tuple): Shape of EEG input data (channels, time points, 1).
            learning_rate (float): Learning rate for the Adam optimizer.
            lstm_units (int): Number of units in the LSTM layer.
            dropout_rate (float): Dropout rate to reduce overfitting.
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.build_cnn_bilstm()

    def build_cnn_bilstm(self):
        """
        Builds a CNN-BiLSTM architecture:
        - 3 convolutional layers for feature extraction (64, 128, 256 filters)
        - Global average pooling to reduce dimensionality
        - Reshape into sequence format for LSTM
        - BiLSTM layer to capture temporal dependencies
        - Dense layer for binary classification
        """
        model = Sequential()

        # Convolution Block 1
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Convolution Block 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Convolution Block 3 + Global Pooling
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())  # Reduces to vector of shape (256,)

        # Reshape for LSTM input: sequence length = 1, feature = 256
        model.add(Reshape((1, 256)))

        # Bidirectional LSTM
        model.add(Bidirectional(LSTM(self.lstm_units, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)))

        # Output layer: Binary classification (sigmoid activation)
        model.add(Dense(1, activation='sigmoid'))

        return model

    def compile_model(self):
        """
        Compiles the model using binary cross-entropy loss and Adam optimizer.
        Tracks recall and precision during training.
        """
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy', Recall(), Precision()])

    def train(self, train_data, train_labels, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trains the CNN-BiLSTM model and plots recall across epochs.

        Parameters:
            train_data (ndarray): Training EEG data.
            train_labels (ndarray): Binary labels.
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.
            validation_split (float): Proportion of training data used for validation.
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(train_data, train_labels, epochs=epochs,
                                 batch_size=batch_size, validation_split=validation_split,
                                 callbacks=[early_stop])

        # Plot training recall (or fallback if unavailable)
        if 'recall' in history.history:
            recall_train = history.history['recall']
            plot_label = "recall"
        elif 'val_recall' in history.history:
            recall_train = history.history['val_recall']
            plot_label = "val_recall"
        else:
            recall_train = history.history['val_loss']
            plot_label = "val_loss"
            print("Available keys in history.history:", history.history.keys())

        # Plotting
        epochs_range = range(1, len(recall_train) + 1)
        plt.plot(epochs_range, recall_train, label=plot_label)
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Training Recall vs Epochs')
        plt.show()

        return history

    def evaluate(self, test_data, test_labels):
        """
        Evaluates model performance and prints accuracy, recall, precision.

        Also plots confusion matrix to visualize true/false positives/negatives.

        Parameters:
            test_data (ndarray): Test EEG data.
            test_labels (ndarray): True binary labels.
        """
        loss, accuracy, recall, precision = self.model.evaluate(test_data, test_labels)
        print(f'Loss: {loss:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}')

        # Generate binary predictions
        y_pred = (self.model.predict(test_data) > 0.5).astype(int)

        # Plot Confusion Matrix
        cm = confusion_matrix(test_labels, y_pred)
        cm_labels = ['Non-seizure', 'Seizure']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

        return loss, accuracy, recall, precision


class EdfToNpy:
    # Constants
    WINDOW_TIME = 10  # segment size in seconds
    STEP_TIME = 5     # Step size in seconds
    SEIZURE_PROPORTION = 0.005    # proportion of non-seizure
    TO_MICROVOLTS = 1e6
    channel_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                      'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                      'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']

    def __init__(self, folder, save_to):
        self.save_to = save_to
        self.folder = folder

    def bandpass_filter(self, data, lowcut=0.5, highcut=50.0, fs=256.0, order=5):
        """
        Applies a bandpass filter to the input data.

        Parameters:
        data (array): The input EEG signal data.
        lowcut (float): The lower cutoff frequency for the filter in Hz.
        highcut (float): The upper cutoff frequency for the filter in Hz.
        fs (float): The sampling frequency of the data in Hz.
        order (int): The order of the Butterworth filter.

         Returns:
        array: The filtered EEG data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def read_edf(self):
        """
        Processes EDF files to extract EEG data and calculate seizure proportions.

        Returns:
        count (int): The total number of EEG windows processed.
        num_channels (int): The total number of EEG channels.
        window_size (int): The size of the window in data points.
        """
        count = 0  # initialize the count variable
        window_size = 0  # initialize the window_size variable

        for file in self.folder:
            edf_data = mne.io.read_raw_edf(file, preload=False)
            edf_labels = edf_data.ch_names

            # Validate channel labels

            if sum([any([0 if re.match(c, l) is None else 1 for l in edf_labels]) for c in EdfToNpy.channel_labels]) == len(EdfToNpy.channel_labels):
                sampling_freq = int(1 / (edf_data.times[1] - edf_data.times[0]))
                window_size = sampling_freq * EdfToNpy.WINDOW_TIME
                window_stride = sampling_freq * EdfToNpy.STEP_TIME

                # Seizure annotation handling

                has_seizure = np.zeros((edf_data.n_times,))
                if os.path.exists(file + '.seizures'):
                    has_annotation = wfdb.rdann(file, 'seizures')
                    for idx in range(int(has_annotation.sample.size / 2)):
                        has_seizure[has_annotation.sample[idx * 2]:has_annotation.sample[idx * 2 + 1]] = 1

                # Calculate seizure proportions in sliding windows

                has_seizure_idx = np.array([has_seizure[idx * window_stride:idx * window_stride + window_size].sum() / window_size for idx in range((edf_data.n_times - window_size) // window_stride)])

                # Calculate non-seizure and seizure window counts

                noseizure_n_size = round(EdfToNpy.SEIZURE_PROPORTION * np.where(has_seizure_idx == 0)[0].size)
                seizure_n_size = np.where(has_seizure_idx > 0)[0].size
                count = count + noseizure_n_size + seizure_n_size  # increment count to tally total samples

            edf_data.close()

        return count, len(EdfToNpy.channel_labels), window_size

    def extract_edf(self, n_samples, n_channel_labels, window_size):
        """
        Extracts EEG signals and labels from EDF files, processes them, and saves the data as NumPy arrays.

        Parameters:
        n_samples (int): The total number of samples to process.
        n_channel_labels (int): The number of EEG channels.
        window_size (int): The size of the window in data points.

        Saves:
        _signals.npy: Processed EEG signals.
        _labels.npy: Corresponding seizure and non-seizure labels.
        """
        signals_np = np.zeros((n_samples, n_channel_labels, window_size), dtype=np.float32)
        labels_np = np.zeros(n_samples, dtype=np.int32)
        count = 0  # initialize count variable

        for number, file in enumerate(self.folder):
            edf_data = mne.io.read_raw_edf(file, preload=False)

            # Check for matching channel labels

            n_label_match = sum([any([0 if re.match(ch, ch_name) is None else 1 for ch_name in edf_data.ch_names]) for ch in EdfToNpy.channel_labels])
            if n_label_match == len(EdfToNpy.channel_labels):
                # Rename channels to standardised labels
                dict_ch_name = {sorted([ch_name for ch_name in edf_data.ch_names if re.match(ch, ch_name) is not None])[0]: ch for ch in EdfToNpy.channel_labels}
                edf_data.rename_channels(dict_ch_name)

                has_seizure = np.zeros((edf_data.n_times,))
                signals_ = edf_data.get_data(picks=EdfToNpy.channel_labels) * EdfToNpy.TO_MICROVOLTS

                # Apply bandpass filter to clean the signal
                signals_ = np.array([self.bandpass_filter(signal) for signal in signals_])

                # Process seizure annotations if available

                if os.path.exists(file + '.seizures'):
                    has_annotation = wfdb.rdann(file, 'seizures')
                    for idx in range(int(has_annotation.sample.size / 2)):
                        has_seizure[has_annotation.sample[idx * 2]:has_annotation.sample[idx * 2 + 1]] = 1

                # Calculate seizure proportions and windows

                sampling_freq = int(1 / (edf_data.times[1] - edf_data.times[0]))
                window_size = sampling_freq * EdfToNpy.WINDOW_TIME
                window_stride = sampling_freq * EdfToNpy.STEP_TIME
                has_seizure_idx = np.array([has_seizure[idx * window_stride:idx * window_stride + window_size].sum() / window_size for idx in range((edf_data.n_times - window_size) // window_stride)])

                # Select random non-seizure windows

                noseizure_n_size = round(EdfToNpy.SEIZURE_PROPORTION * np.where(has_seizure_idx == 0)[0].size)

                # Non-seizure data (random sampling)
                temp_negative = random.sample(list(np.where(has_seizure_idx == 0)[0]), noseizure_n_size)
                for value in temp_negative:
                    start_index = value * window_stride
                    stop_index = value * window_stride + window_size
                    signals_np[count, :, :] = signals_[:, start_index:stop_index]
                    labels_np[count] = 0
                    count = count + 1

                # Seizure data
                temp_positive = list(np.where(has_seizure_idx > 0)[0])
                for value in temp_positive:
                    start_index = value * window_stride
                    stop_index = value * window_stride + window_size
                    signals_np[count, :, :] = signals_[:, start_index:stop_index]
                    labels_np[count] = 1
                    count = count + 1
            else:
                print(f"Unable to read {file}")

            edf_data.close()

        # Save the processed data
        np.save(self.save_to + '_signals', signals_np)
        np.save(self.save_to + '_labels', labels_np)

    def show_eeg(self, signals, label=None):
        """
        Visualises EEG signals by plotting each channel vertically with an offset for easier differentiation.

        Parameters:
        signals (array): A NumPy array containing EEG signal data
        """
        # Define vertical spacing between channels in the plot
        vertical_width = 250

        # Sampling frequency of the EEG signal in Hz
        fs = 256

        # Create a new figure and axis for the plot

        fig, ax = plt.subplots()
        for i in range(signals.shape[0]):
            # Plot the signal for the current channel with time on the x-axis
            ax.plot(np.arange(signals.shape[-1]) / fs, signals[i, :] + i * vertical_width, linewidth=0.5, color='tab:blue')
            # Annotate the channel label next to its corresponding signal
            ax.annotate(EdfToNpy.channel_labels[i], xy=(0, i * vertical_width))

        # Add plot title based on label
        if label is not None:
            title = "Seizure" if label == 1 else "Non-Seizure"
            ax.set_title(f"EEG Sample - {title}", fontsize=14)

        # Invert the y-axis so that the topmost channel appears first
        ax.invert_yaxis()
        plt.xlabel("Time (s)")
        plt.show()

    class SeizurePredictor:
        """
        Predict seizures from EEG segments using a trained CNN-BiLSTM model.

        Supports both:
        - Batch prediction on multiple EEG segments.
        - Real-time single-segment prediction with optional visualization.

        Attributes:
            model (tf.keras.Model): Loaded CNN-BiLSTM model.
            plotter (EdfToNpy): Utility for EEG visualization.
        """

        def __init__(self, model_path):
            """
            Load a trained CNN-BiLSTM model for inference.

            Parameters:
                model_path (str): Path to the saved Keras H5 model.
            """
            self.model = load_model(model_path, compile=False)
            self.plotter = EdfToNpy([], save_to=None)

        def preprocess(self, signal):
            """
            Apply bandpass filter, z-score normalization, and reshape.

            Parameters:
                signal (np.ndarray): EEG signal(s) with shape (N, 23, 2560)

            Returns:
                np.ndarray: Preprocessed data of shape (N, 23, 2560, 1)
            """
            filtered, _ = bandpass_filter(signal), None
            normalized, _ = normalise_data(filtered)
            reshaped = normalized[..., np.newaxis]
            return reshaped

        def predict(self, signal):
            """
            Predict seizure labels for a batch of EEG signals.

            Parameters:
                signal (np.ndarray): Input shape (N, 23, 2560)

            Returns:
                np.ndarray: Binary predictions (0 or 1) for each sample.
            """
            reshaped = self.preprocess(signal)
            predictions = (self.model.predict(reshaped) > 0.5).astype(int)
            return predictions

        def predict_one(self, signal):
            """
            Predict seizure status for a single EEG segment.

            Parameters:
                signal (np.ndarray): Must be shape (1, 23, 2560)

            Returns:
                int: 1 if seizure, 0 otherwise.

            Raises:
                ValueError: If input shape is invalid (e.g., (23, 2560))
            """
            if signal.ndim != 3 or signal.shape[0] != 1:
                raise ValueError(
                    f"Invalid input shape {signal.shape}. "
                    "Expected shape: (1, 23, 2560). Wrap input in [ ] if needed."
                )

            reshaped = self.preprocess(signal)
            prediction = (self.model.predict(reshaped) > 0.5).astype(int)
            return prediction[0][0]

        def display_seizures(self, signal, predictions):
            """
            Visualize all segments predicted as seizures.

            Parameters:
                signal (np.ndarray): Raw EEG data, shape (N, 23, 2560)
                predictions (np.ndarray): Binary prediction array (N,)
            """
            seizure_indices = np.where(predictions.ravel() == 1)[0]
            if len(seizure_indices) == 0:
                print("No seizure segments detected.")
                return

            print(f"Displaying {len(seizure_indices)} predicted seizure segment(s):")
            for idx in seizure_indices:
                self.plotter.show_eeg(signal[idx], label=f"Predicted Seizure (Index {idx})")
