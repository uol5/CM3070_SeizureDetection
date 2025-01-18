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

#### Caching
Wavelet and PSD preprocessing takes some time. Therefore, it makes sense to cache the results for efficiency.

In-built 'lre-cache' requires hashable input. Arrays have to be converted to hashable format, such as a tuple, using tuple(numpy array). This consumes memory and computation resource.

Since there are few but very large arrays to store in this programme, use of python dictionary is more efficient. Hashing is not necessary, as key name lookup for so few arrays are sufficiently efficient.

cache_decorator is implemented as a decorator. Arrays are cached as simple name-value pairs in cache_dictionary inside SVN_Tools.



class SVM_Tools:
    def __init__(self,data, labels):
        self.full_data = data
        self.full_labels = labels
        self.cache = {}
        self.wavelet = self.cache_decorator("wavelet")(self.get_wavelet)
        self.get_psd = self.cache_decorator("psd")(self.get_psd)

    # build and train SVM model on the CHB-MIT data signals.
    @staticmethod
    def train_svm(data, labels):
        if len(data.shape) == 2:
            X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, test_size=0.3)
        elif len(data.shape) == 3:
            n_samples, timesteps, features = data.shape
            data_reshaped = np.reshape(data, (n_samples,timesteps*features))
            X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels, train_size=0.7, test_size=0.3)
        else:
            print("Incompatible data shape")
            return

       # params = { "kernel": ["linear", "rbf"],
       #            "C": [1, 10, 100],              # Regularization parameter
       #            "gamma": [0.01, 0.1, 1, 10, 50] } # Kernel coefficient for 'rbf'
        params = { "kernel": ["linear"],
                   "C": [1],              # Regularization parameter
                   "gamma": [5] } # Kernel coefficient for 'rbf'


        # Coding technique for handling training failure
        try:
            svm = SVC(probability=True, verbose=1)

            # RandomizedSearchCV is compute intensive, use RandomSearchCv to speed up search for optimal parameters
            clf = RandomizedSearchCV(svm,param_distributions=params,cv=2,n_jobs=-1,n_iter=20)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #svm = SVC(C=1.0, gamma='scale', kernel='rbf', probability=True, verbose=1)
            clf.fit(X_train_scaled, y_train )
            y_predict = clf.predict(X_test_scaled)

            print(f"Best parameters set: {clf.best_params_}")
            print("\nConfusion matrix:")
            print(confusion_matrix(y_test, y_predict))
            print("\nClassification report:")
            print(classification_report(y_test, y_predict))

            return clf


            #clf = SVC(C=1.0, gamma='scale', probability=True, verbose=1)
            #clf.fit(X_train, y_train)
            #y_predict = clf.predict(X_test)




            # print("\nConfusion matrix:")
            # print(confusion_matrix(y_test, y_predict, labels=labels))
            # print("\nClassification report:")
            # print(classification_report(y_test, y_predict, labels=labels))
            #
            # return clf
        # Coding technique to allow fallback when failure due large sample size
        except MemoryError as e:
            small_sample = 1000
            print(f"Insufficient memory for SVM: {e}")
            print(f"Retrain using first {small_sample} samples,  C = 1.0, Gamma ='scale'(auto adjust to feature size)")
            # Initialise simple SVC model with common used parameters for  C and gamma
            simple_svm = SVC(kernel='linear', C=1.0, gamma='scale', probability=True, verbose=1)
            # Perform 2 fold cross validation
            cv_score = cross_val_score(simple_svm, X_train[:small_sample], y_train[:small_sample], cv=2, n_jobs=1, verbose=1)
            # Print score
            print(f"Cross validation score: {cv_score}")
            print(f"Average cross validation score: {cv_score.mean()}")
        # Error handling for all other errors
        except Exception as e:
            print(f"Failure during SVM training with error {e}")

    # feature to be extracted is the power spectral density for each channel
    def get_psd(self, frequency=256, segment_length=256) -> np.ndarray:
        # sample size too large for single step processing. Resulted in out of memory error.
        # Batch process data instead.
        # Try-Except block forms coding technique to gracefully handle error
        try:
            psd_features = []
            for channel in self.full_data:
                freq, psd = welch(x=channel, fs=frequency, nperseg=segment_length)
                psd_features.append(psd)
            return np.array(psd_features)
        except Exception as e:
            print(f"Error in calculating power spectral dnesity: {e}")
            return np.array([]) # return empty numppy array

    # Coding technique to optimise real time data handling and memory efficiency by batch processing.
    def get_psd_batch(self, batch_size=1000) -> np.ndarray:
        batch_size = 100
        full_features = []
        #pass batch size of samples to get_psd. Store it and concatenate all obtained result into single numpy array.
        for start in range(0,self.full_data.shape[1],batch_size):
            end = min(start + batch_size, self.full_data.shape[1])
            each_batch = self.full_data[:,start:end]
            batch_psd = self.get_psd(each_batch)
            full_features.append(batch_psd)

        # each batch is appended as a numpy array into full_features. Concatenate the batches into a single batch (row wise).
        full_features = np.concatenate(full_features, axis = 1)
        return full_features

    @staticmethod
    def to_svm_input(data_signal):
        """convert to n_samples x n_features which is required for SVM model training"""
        if data_signal.ndim == 2:
            return
        elif data_signal.ndim > 3:
            # flatten to 2-D shape of (number samples, number features).  i.e. (number samples , channels X datapoints)
            data_signal.reshape(data_signal,-1)
        return data_signal

    def show_shape(self):
       return self.full_data.shape

    def get_wavelet(self, save_to = 'waveletSVM.csv', scales=np.arange(1,30), wavelet='cmor',batch_size=50) -> np.ndarray:
        # results of wavelet transforms take 20 minutes to run. Save to file waveletSVM.csv
        if os.path.exists(save_to):
            wav_temp = np.loadtxt(save_to, delimiter=',')
            print("wavelet transformed file already exists. Import from file")
            return np.array(wav_temp)
        # useful brain activity 1 - 50 Hz
        # apply band filter. Remove > 50 Hz, and < 1 Hz frequencies
        # Nyquist frequency is twice the frequency that can be reconstituted from data
        # parameters to create filter using butter function from scipy.signal
        sampling_frequency = 256  # Sampling frequency from CHB-MIT dataset
        nyquist = 0.5 * sampling_frequency
        low_hz = 1
        high_hz = 50
        order = 4  # scipy.signal.butter documentation recommends 4 for bandpass

        # User Butterworth filter to create bandpass filter
        lower_limit = low_hz / nyquist
        higher_limit = high_hz / nyquist
        b,a = butter(order, [lower_limit, higher_limit], btype='band')
        # returns (numerator, denominator)

        # Apply Butterworth filter to filter data
        # returns same shaped numpy array with > 50 & < 1 values filtered out
        filtered_signals = filtfilt(b,a,self.full_data,axis =-1)
        n_samples, n_channels, n_datapoints = filtered_signals.shape

        samples_features = [] # list of features for each sample
        # for each sample, extract wavelet features for each chanel, and find mean of all channels in a sample
        for sample in range(n_samples):
            channel_features = [] # list of features for each channel
            for channel in range(n_channels):
                # perform Morlet wavelet transform using pywt library for each channel separately
                # cwt is continuo  wavelet transformation (for continuous variable)
                # 128 different scales ( "resolutions") applied using np.arange(1,128)
                coefficient, freq = pywt.cwt(filtered_signals[sample, channel], scales, wavelet)

                # coefficient are complex numbers, may be positive or negative. Squared value removes negative numbers
                # np.abs(coefficient) returns magnitude  np.abs(coefficient)**2 returns power (intensity or energy in different frequency band).
                power = np.abs(coefficient)**2

                # take mean over time axis
                feature = np.mean(power, axis = 1)
                channel_features.append(feature)

            # concatenate all channel features for this sample
            samples_features.append(np.concatenate(channel_features, axis=0))  # alternative, np.stack may be useful to create new axis
        np_features = np.array(samples_features)
        # coding technique for interoperability by saving output in standard format such as CSV.
        np.savetxt(save_to, np_features, delimiter =',')

        return np_features

    # In-build lre-cache requires hashable input. Storing post processed numpy array will require converting to a hashable form, such as a tuple using  tuple(numpy array).
    # Since there are limited number of arrays , it is  more efficient to use python dictionary.
    def cache_decorator(self,name):
        def inside_decorator(func):
            # calls nested function
            def wrapper(*args,**kwargs):
                array_result = func(*args,**kwargs)
                try:
                    if not isinstance(array_result, np.ndarray):
                        array_result = np.array(array_result)
                except TypeError:
                    print("Wrong data type")
                # store key-value pair in cache_dict
                self.cache[name] = array_result
                return array_result
            return wrapper
        return inside_decorator

    # Coding technique to check input data validity - 1) arrays are numpy and 2) data and label samples match
    def validate_data(self, data, labels):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data should be a NumPy array.")
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels should be a NumPy array.")
        if data.shape[0] != len(labels):
            raise ValueError("Number of samples in data must match number of labels.")

    #########################################################
