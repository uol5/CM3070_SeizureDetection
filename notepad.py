from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, filtfilt, welch
import numpy as np
import os
import pywt
from sklearn.preprocessing import StandardScaler

class SVMClassifier:
    def __init__(self, data, label, kernel='rbf', random_state=42):
        self.kernel = kernel
        self.random_state = random_state
        self.model = SVC(kernel=self.kernel, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.train_signals = self.bandpass_filter(data)
        self.train_labels = label  # Labels don't require filtering
        self.train_signals = self.scale_features(self.train_signals)
        self.feature = None

    def bandpass_filter(self, data, low_hz=1, high_hz=50, sampling_frequency=256, order=4):
        nyquist = 0.5 * sampling_frequency
        low = low_hz / nyquist
        high = high_hz / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def scale_features(self, data):
        """
        Scales the train_signals to have zero mean and unit variance.
        Assumes data shape is (samples, channels, datapoints).
        """
        n_samples, n_channels, n_datapoints = data.shape
        data_reshaped = data.reshape(n_samples, -1)  # Flatten channels and datapoints
        scaled_data = self.scaler.fit_transform(data_reshaped)  # Scale using StandardScaler
        return scaled_data.reshape(n_samples, n_channels, n_datapoints)  # Reshape back to original shape

    def get_psd_features(self, data, sampling_frequency=256, save_to='psd_features.csv'):
        if save_to and os.path.exists(save_to):
            print("PSD features already exist. Loading from file.")
            return np.loadtxt(save_to, delimiter=',')

        n_samples, n_channels, n_datapoints = data.shape
        psd_features = []

        for sample in range(n_samples):
            sample_features = []
            for channel in range(n_channels):
                freqs, psd = welch(data[sample, channel], fs=sampling_frequency, nperseg=256)
                sample_features.append(psd)
            psd_features.append(np.concatenate(sample_features))

        psd_features = np.array(psd_features)
        if save_to:
            np.savetxt(save_to, psd_features, delimiter=',')

        self.feature = psd_features
        return psd_features

    def get_wavelet_features(self, data, scales=np.arange(1, 30), wavelet='cmor1.5-1.0', save_to='wavelet_features.csv') -> np.ndarray:
        if save_to and os.path.exists(save_to):
            print("Wavelet features already exist. Loading from file.")
            return np.loadtxt(save_to, delimiter=',')

        n_samples, n_channels, n_datapoints = data.shape
        wavelet_features = []

        for sample in range(n_samples):
            sample_features = []
            for channel in range(n_channels):
                coeffs, _ = pywt.cwt(data[sample, channel], scales, wavelet)
                power = np.abs(coeffs) ** 2
                mean_power = np.mean(power, axis=1)
                sample_features.append(mean_power)
            wavelet_features.append(np.concatenate(sample_features))

        wavelet_features = np.array(wavelet_features)
        if save_to:
            np.savetxt(save_to, wavelet_features, delimiter=',')

        self.feature = wavelet_features
        return wavelet_features

    def parameter_tuning(self, features, labels, param_grid, cv=3):
        grid_search = GridSearchCV(SVC(random_state=self.random_state), param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(features, labels)
        print("Best parameters found:", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def train_and_evaluate(self, features, labels, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report



if __name__ == "__main__":
    import os

    # Load your data (replace with actual file paths or data loaders)
    # Assuming train_signals_1s, train_labels_1s, train_signals_10s, and train_labels_10s are already loaded

    sampling_frequency = 256
    low_hz, high_hz = 1, 50
    scales = np.arange(1, 30)
    wavelet = 'cmor'

    classifier = SVMClassifier(kernel='rbf', random_state=42)

    # Apply bandpass filter
    train_signals_1s_filtered = classifier.bandpass_filter(train_signals_1s, low_hz, high_hz, sampling_frequency)
    train_signals_10s_filtered = classifier.bandpass_filter(train_signals_10s, low_hz, high_hz, sampling_frequency)

    # PSD Features
    psd_features_1s = classifier.get_psd_features(train_signals_1s_filtered, sampling_frequency, save_to='psd_features_1s.csv')
    psd_features_10s = classifier.get_psd_features(train_signals_10s_filtered, sampling_frequency, save_to='psd_features_10s.csv')

    # Wavelet Features
    wavelet_features_1s = classifier.get_wavelet_features(train_signals_1s_filtered, scales, wavelet, save_to='wavelet_features_1s.csv')
    wavelet_features_10s = classifier.get_wavelet_features(train_signals_10s_filtered, scales, wavelet, save_to='wavelet_features_10s.csv')

    # Train and evaluate SVM
    print("\n--- PSD Features (1-second window) ---")
    psd_accuracy_1s, psd_report_1s = classifier.train_and_evaluate(psd_features_1s, train_labels_1s)
    print(f"Accuracy: {psd_accuracy_1s}\n{psd_report_1s}")

    print("\n--- PSD Features (10-second window) ---")
    psd_accuracy_10s, psd_report_10s = classifier.train_and_evaluate(psd_features_10s, train_labels_10s)
    print(f"Accuracy: {psd_accuracy_10s}\n{psd_report_10s}")

    print("\n--- Wavelet Features (1-second window) ---")
    wavelet_accuracy_1s, wavelet_report_1s = classifier.train_and_evaluate(wavelet_features_1s, train_labels_1s)
    print(f"Accuracy: {wavelet_accuracy_1s}\n{wavelet_report_1s}")

    print("\n--- Wavelet Features (10-second window) ---")
    wavelet_accuracy_10s, wavelet_report_10s = classifier.train_and_evaluate(wavelet_features_10s, train_labels_10s)
    print(f"Accuracy: {wavelet_accuracy_10s}\n{wavelet_report_10s}")


########################################################################################################