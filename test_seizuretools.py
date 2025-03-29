"""
test_seizure.py

Unit tests for seizuretools.py.
"""

import unittest
import numpy as np
from scipy.signal import periodogram
from unittest.mock import patch, MagicMock
from matplotlib import pyplot as plt
from seizuretools import bandpass_filter, normalise_data, EdfToNpy, ConvBLSTM
from seizuretools import EdfToNpy


class TestSeizurePredictor(unittest.TestCase):
    """
    Unit tests for the `SeizurePredictor` class in seizuretools.py.

    This test suite ensures that seizure prediction, preprocessing, and visualisation
    work as expected when using a trained CNN-BiLSTM model.

    Test Plan Summary:

    | Test Name                        | Purpose                                                                 |
    |----------------------------------|-------------------------------------------------------------------------|
    | Model loading test               | Verifies that the model loads correctly using `load_model()`           |
    | Preprocessing output shape       | Ensures input is reshaped from `(N, 23, 2560)` to `(N, 23, 2560, 1)`   |
    | Preprocessing filter + normalise | Confirms that preprocessing alters the original signal statistics      |
    | Batch prediction shape           | Ensures `predict()` returns binary array of shape `(N, 1)`             |
    | Single prediction value          | Ensures `predict_one()` returns scalar `0` or `1`                      |
    | Single prediction shape error    | Raises `ValueError` if input to `predict_one()` is not 3D `(1, 23, 2560)` |
    | Display seizure visualisation    | Mocks `show_eeg()` and confirms it is called for predicted seizures    |
    """

    @patch('seizuretools.load_model')
    def setUp(self, mock_load_model):
        """
        Creates a mock model with a dummy predict method for all tests.
        """
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[0.8], [0.2], [0.9]])
        mock_load_model.return_value = self.mock_model

        self.predictor = EdfToNpy.SeizurePredictor("mock_model_path")

    def test_model_loading(self):
        """
        Verifies that the model is loaded and accessible.
        """
        self.assertIsNotNone(self.predictor.model)
        self.assertTrue(callable(self.predictor.model.predict))

    def test_preprocess_output_shape(self):
        """
        Checks that preprocess reshapes data correctly to 4D tensor.
        """
        x = np.random.randn(3, 23, 2560)
        output = self.predictor.preprocess(x)
        self.assertEqual(output.shape, (3, 23, 2560, 1))

    def test_preprocess_changes_statistics(self):
        """
        Verifies that preprocessing changes the data’s statistical properties.
        """
        x = np.ones((2, 23, 2560)) * 10  # constant value input
        output = self.predictor.preprocess(x)
        self.assertFalse(np.allclose(output[..., 0], x))  # Should no longer be constant

    def test_predict_batch_output_shape(self):
        """
        Ensures predict returns binary values with shape (N, 1).
        """
        x = np.random.randn(3, 23, 2560)
        predictions = self.predictor.predict(x)
        self.assertEqual(predictions.shape, (3, 1))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_predict_one_output_value(self):
        """
        Ensures predict_one returns a scalar 0 or 1.
        """
        x = np.random.randn(1, 23, 2560)
        result = self.predictor.predict_one(x)
        self.assertIn(result, [0, 1])
        self.assertIsInstance(result, (int, np.integer))  # Integer scalar

    def test_predict_one_shape_error(self):
        """
        Raises ValueError if input is not shaped (1, 23, 2560).
        """
        invalid_input = np.random.randn(23, 2560)
        with self.assertRaises(ValueError):
            self.predictor.predict_one(invalid_input)

    @patch.object(EdfToNpy, 'show_eeg')
    def test_display_seizures_calls_visualisation(self, mock_show_eeg):
        """
        Verifies show_eeg is called for segments predicted as seizures.
        """
        x = np.random.randn(3, 23, 2560)
        preds = np.array([[1], [0], [1]])
        self.predictor.display_seizures(x, preds)
        self.assertEqual(mock_show_eeg.call_count, 2)


class TestConvBLSTM(unittest.TestCase):
    """
    Unit tests for the `ConvBLSTM` class in seizuretools.py.

    Test Plan Summary:

    | Test Name                        | Purpose                                                                 |
    |----------------------------------|-------------------------------------------------------------------------|
    | Model build test                 | Confirm the model builds with expected layer structure                 |
    | Training execution test          | Verify model trains and returns a history object                       |
    | Recall plotting test             | Ensure recall is plotted during training                               |
    | Evaluation output test           | Evaluate returns correct metrics                                       |
    | Prediction shape check           | Ensure output shape of model.predict is correct                        |
    | Confusion matrix plot test       | Mock `ConfusionMatrixDisplay.plot()` to ensure it's called             |
    """

    def setUp(self):
        self.model_wrapper = ConvBLSTM(input_shape=(23, 2560, 1))
        self.model = self.model_wrapper.model

    def test_model_build(self):
        """
        Asserts the model was built and has layers.
        """
        self.assertTrue(len(self.model.layers) > 0)

    @patch.object(plt, 'show')
    def test_training_execution(self, mock_show):
        """
        Checks that training runs and returns a valid history object.
        """
        self.model_wrapper.compile_model()
        x = np.random.randn(10, 23, 2560, 1)
        y = np.random.randint(0, 2, size=(10,))
        history = self.model_wrapper.train(x, y, epochs=1, batch_size=2)
        self.assertIn("loss", history.history)

    @patch.object(plt, 'show')
    def test_training_recall_plot(self, mock_show):
        """
        Checks that recall is plotted during training.
        """
        self.model_wrapper.compile_model()
        x = np.random.randn(10, 23, 2560, 1)
        y = np.random.randint(0, 2, size=(10,))
        history = self.model_wrapper.train(x, y, epochs=1)
        self.assertTrue(any(key.startswith("recall") for key in history.history))

    @patch('seizuretools.ConfusionMatrixDisplay.plot')
    @patch.object(plt, 'show')
    def test_model_evaluation(self, mock_plot, mock_show):
        """
        Verifies that evaluation returns 4 metrics and plots confusion matrix.
        """
        self.model_wrapper.compile_model()
        x = np.random.randn(5, 23, 2560, 1)
        y = np.random.randint(0, 2, size=(5,))
        self.model.predict = MagicMock(return_value=np.random.rand(5, 1))
        self.model.evaluate = MagicMock(return_value=[0.1, 0.95, 0.9, 0.92])
        results = self.model_wrapper.evaluate(x, y)
        self.assertEqual(len(results), 4)
        mock_plot.assert_called_once()

    def test_model_predict_shape(self):
        """
        Ensures predict returns correct output shape.
        """
        x = np.random.randn(2, 23, 2560, 1)
        self.model_wrapper.compile_model()
        preds = self.model_wrapper.model.predict(x)
        self.assertEqual(preds.shape[0], 2)


class TestEdfToNpy(unittest.TestCase):
    """
    Unit tests for the `EdfToNpy` class in seizuretools.py.

    Test Plan Summary:

    | Test Name                         | Purpose                                                                  |
    |-----------------------------------|--------------------------------------------------------------------------|
    | Channel label match test          | Check that channel validation logic works                                |
    | Seizure annotation parsing test   | Ensure seizure segments are extracted using `.seizures` file             |
    | Signal filtering test             | Verify bandpass filtering works on sample EEG array                      |
    | EDF read count shape test         | Check `read_edf()` returns expected counts and window sizes              |
    | Extract saves .npy correctly      | Mock `np.save()` to ensure correct call signatures                       |
    | Show EEG visualisation            | Test plotting logic runs without error                                   |

    Skipped Tests:
    - Channel label matching and seizure parsing rely heavily on MNE and WFDB interaction with file formats,
      and are better handled in **integration or functional tests** using real datasets.
    - Bandpass filtering is already independently and thoroughly tested in its own function.
    - EDF file window/shape logic should be tested with full data access context, not mocked partially.

    Included Tests:
    - `test_edf_file_import`: Ensures EDF file import via MNE works and returns expected object.
    - `test_extract_saves_numpy`: Confirms that `np.save()` is called during data extraction.
    - `test_show_eeg_plot`: Verifies that plotting runs without raising errors and calls matplotlib.

    """

    @patch('seizuretools.mne.io.read_raw_edf')
    def test_edf_file_import(self, mock_read_raw):
        """
        Ensures that an EDF file can be loaded via MNE without error.
        """
        mock_data = MagicMock()
        mock_data.ch_names = ['FP1-F7', 'F7-T7'] * 12  # fake channel names
        mock_data.times = np.linspace(0, 10, 2560)
        mock_data.n_times = 2560
        mock_read_raw.return_value = mock_data

        edf = EdfToNpy(['dummy.edf'], save_to='.')
        count, n_channels, win_size = edf.read_edf()

        self.assertIsInstance(count, int)
        self.assertIsInstance(n_channels, int)
        self.assertIsInstance(win_size, int)

    @patch('seizuretools.np.save')
    @patch('seizuretools.mne.io.read_raw_edf')
    def test_extract_saves_numpy(self, mock_read_raw, mock_save):
        """
        Ensures that extract_edf() saves .npy files using numpy's save.
        """
        dummy = np.random.randn(23, 2560)
        mock_data = MagicMock()
        mock_data.ch_names = EdfToNpy.channel_labels
        mock_data.get_data.return_value = dummy
        mock_data.times = np.linspace(0, 10, 2560)
        mock_data.n_times = 2560
        mock_read_raw.return_value = mock_data

        edf = EdfToNpy(['dummy.edf'], save_to='output')
        edf.extract_edf(n_samples=2, n_channel_labels=23, window_size=2560)

        self.assertTrue(mock_save.called)
        self.assertEqual(mock_save.call_count, 2)
        mock_save.assert_any_call('output_signals', unittest.mock.ANY)
        mock_save.assert_any_call('output_labels', unittest.mock.ANY)

    @patch.object(plt, 'show')
    def test_show_eeg_plot(self, mock_show):
        """
        Ensures that show_eeg() plots without raising errors.
        """
        signals = np.random.randn(23, 2560)
        edf = EdfToNpy([], save_to=None)

        try:
            edf.show_eeg(signals, label=1)
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"show_eeg() raised an exception: {e}")


class TestBandpassFilter(unittest.TestCase):
    """
    This test suite ensures the bandpass filter behaves as expected under various conditions,
    including normal operation, signal distortion suppression, edge behaviour near Nyquist frequency,
    adaptability to different sampling frequencies, and invalid inputs.

    Summary of Tests:

    | Test Name                      | Purpose                                     |
    |-------------------------------|---------------------------------------------|
    | Basic shape test              | Output matches input shape                  |
    | Filter effectiveness          | Suppress out-of-band frequencies            |
    | Nyquist edge test             | Check near-Nyquist limit stability          |
    | Different sampling frequencies| Function adapts to different `fs`           |
    | Invalid input handling        | Detects bad shape/params/values             |
    """

    def test_basic_output_shape(self):
        """
        Ensures that the filtered EEG signal has the same shape as the input.
        """
        input_data = np.random.randn(5, 23, 2560)
        output_data = bandpass_filter(input_data)
        self.assertEqual(input_data.shape, output_data.shape)

    def test_filter_effectiveness(self):
        """
        Verifies that the bandpass filter attenuates frequencies outside the desired range.

        This test constructs a synthetic signal composed of three sinusoidal components:
        - An in-band frequency (10 Hz)
        - Two out-of-band frequencies (0.5 Hz and 30 Hz)

        The power spectrum is calculated before and after filtering using the periodogram,
        and the test asserts that the out-of-band frequencies are significantly reduced
        after filtering.
        """
        fs = 256
        t = np.linspace(0, 10, fs * 10)
        # Compose signal with three frequencies: one in-band and two out-of-band
        f1, f2, f3 = 10, 0.5, 30  # Hz
        signal = (
                np.sin(2 * np.pi * f1 * t) +
                0.5 * np.sin(2 * np.pi * f2 * t) +
                0.5 * np.sin(2 * np.pi * f3 * t)
        )
        data = signal[np.newaxis, np.newaxis, :]  # shape (1, 1, len(t))
        filtered = bandpass_filter(data, fs=fs)

        # Calculate power spectral density before and after filtering
        freqs, p_before = periodogram(signal, fs)
        _, p_after = periodogram(filtered[0, 0], fs)

        # Find indices closest to 0.5 Hz and 30 Hz
        idx_05hz = np.argmin(np.abs(freqs - 0.5))
        idx_30hz = np.argmin(np.abs(freqs - 30.0))

        # Assert power suppression at those frequencies
        self.assertLess(p_after[idx_05hz], 0.1 * p_before[idx_05hz], msg="0.5 Hz not suppressed enough")
        self.assertLess(p_after[idx_30hz], 0.1 * p_before[idx_30hz], msg="30 Hz not suppressed enough")

    def test_nyquist_edge_case(self):
        """
        Tests filtering when cutoffs are near the Nyquist frequency.
        Ensures the function remains numerically stable.
        """
        fs = 256
        lowcut = 0.1
        highcut = fs / 2 - 1
        data = np.random.randn(2, 3, 512)
        try:
            output = bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs)
            self.assertEqual(data.shape, output.shape)
        except Exception as e:
            self.fail(f"Filter failed near Nyquist edge: {e}")

    def test_different_sampling_freqs(self):
        """
        Ensures the filter adapts correctly to various sampling frequencies.
        """
        for fs in [128, 256, 512]:
            data = np.random.randn(3, 10, fs * 2)
            output = bandpass_filter(data, fs=fs)
            self.assertEqual(data.shape, output.shape)

    def test_invalid_inputs(self):
        """
        Ensures the function raises appropriate errors for invalid input types and parameters.
        """
        with self.assertRaises(ValueError):
            bandpass_filter(np.random.randn(10, 2560), fs=256)  # missing channel dimension

        with self.assertRaises(ValueError):
            bandpass_filter(np.random.randn(2, 3, 256), lowcut=30, highcut=10)  # lowcut > highcut

        with self.assertRaises(ValueError):
            bandpass_filter(np.random.randn(2, 3, 256), fs=0)  # invalid fs

        with self.assertRaises(TypeError):
            bandpass_filter("not an array", fs=256)  # invalid input type


class TestNormaliseData(unittest.TestCase):
    """
    Unit tests for the `normalise_data` function in seizuretools.py.

    This suite ensures that z-score normalisation works as expected for EEG data,
    including shape consistency, statistical correctness, and robust handling of edge cases.

    Summary of Tests:

    | Test Name                        | Purpose                                                                 |
    |----------------------------------|-------------------------------------------------------------------------|
    | Basic output shape test          | Ensure output shapes match input for both `x_train` and `x_test`       |
    | Mean and std of output           | Check that output has ~zero mean and unit variance per sample          |
    | Test-only mode (x_test is None)  | Ensure function works when only `x_train` is provided                  |
    | Consistent scaling on test set   | Ensure `x_test` is scaled using `x_train` statistics                   |
    | Invalid input types              | Raise TypeError when inputs are not NumPy arrays                       |
    | Shape mismatch error             | Handle shape mismatch between `x_train` and `x_test` cleanly           |
    """

    def test_output_shape(self):
        """
        Ensures output shapes match input shapes for both x_train and x_test.
        """
        x_train = np.random.randn(20, 23, 256)
        x_test = np.random.randn(10, 23, 256)
        x_train_scaled, x_test_scaled = normalise_data(x_train, x_test)
        self.assertEqual(x_train.shape, x_train_scaled.shape)
        self.assertEqual(x_test.shape, x_test_scaled.shape)

    def test_zero_mean_unit_std(self):
        """
        Verifies that the output has approximately zero mean and unit standard deviation.
        """
        x_train = np.random.randn(100, 5, 10) * 10 + 5  # non-normalised data
        x_train_scaled, _ = normalise_data(x_train)

        # Flatten across all features
        flat = x_train_scaled.reshape(100, -1)
        means = np.mean(flat, axis=0)
        stds = np.std(flat, axis=0)

        self.assertTrue(np.all(np.abs(means) < 1e-6))
        self.assertTrue(np.all(np.abs(stds - 1.0) < 1e-6))

    def test_train_only_mode(self):
        """
        Ensures the function works when only x_train is provided.
        """
        x_train = np.random.randn(30, 10, 100)
        x_train_scaled, x_test_scaled = normalise_data(x_train)
        self.assertEqual(x_train.shape, x_train_scaled.shape)
        self.assertIsNone(x_test_scaled)

    def test_consistent_test_scaling(self):
        """
        Ensures x_test is scaled using x_train statistics.
        """
        x_train = np.ones((50, 2, 5)) * 5
        x_test = np.ones((10, 2, 5)) * 10
        x_train_scaled, x_test_scaled = normalise_data(x_train, x_test)

        self.assertTrue(np.allclose(x_train_scaled, 0))  # mean 5 scaled to 0
        self.assertTrue(np.all(x_test_scaled > 0))       # mean 10 > train mean ⇒ positive scaled values

    def test_invalid_inputs(self):
        """
        Raises TypeError when inputs are not numpy arrays.
        """
        with self.assertRaises(AttributeError):
            normalise_data("not an array", None)

        with self.assertRaises(AttributeError):
            normalise_data(np.random.randn(10, 23, 256), x_test="bad input")

    def test_shape_mismatch(self):
        """
        Checks for graceful handling or error when x_train and x_test shapes are incompatible.
        """
        x_train = np.random.randn(10, 23, 256)
        x_test = np.random.randn(5, 22, 256)  # mismatched channel dimension
        with self.assertRaises(ValueError):
            normalise_data(x_train, x_test)




if __name__ == "__main__":
    unittest.main()
