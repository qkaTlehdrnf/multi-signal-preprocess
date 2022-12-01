import numpy as np
import torch as tc
from scipy.signal import savgol_filter
import statsmodels.api as sm
from scipy.signal import butter, lfilter
from scipy.stats import norm


def std(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    return np.apply_along_axis(lambda x: (x-mean)/std, 1, data)


def torch_apply_along_axis(function, dim: int, data):
    '''unlike np.apply_along_axis, this function's dim is the dim to apply function, not the dim to apply function on'''
    return tc.stack([
        function(x_i) for x_i in tc.unbind(data, dim=dim)
    ], dim=dim)


class PreprocessingFiltering:
    def __init__(self, data=None):
        self.data = data

    def __type_check__(self):
        return 'preprocessing.filtering'

    def load_data(self, data):
        self.data = data


class CustomMean(PreprocessingFiltering):
    def __init__(self, data=None):
        super().__init__(data)
        self.time_interval = 100

    def __name__(self):
        return 'CustomMean'

    def __condition__(self, time_interval=None):
        if time_interval:
            self.time_interval = time_interval
        return f'{self.__name__}: time_interval = {self.time_interval}'

    def filtering(self, data=None):
        if data is not None:
            self.data = data
        return np.apply_along_axis(lambda x: np.mean(x, axis=0), 0, self.data)


class Lfiltering(PreprocessingFiltering):
    def __init__(self, data=None):
        super().__init__(data)
        self.b, self.a = [1/100]*100, 2

    def __name__(self):
        return 'Lfiltering'

    def __condition__(self, b=None, a=None):
        if b:
            self.b = b
        if a:
            self.a = a
        return f'{self.__name__}: b = {self.b}, a = {self.a}'

    def filtering(self, data=None):
        if data is not None:
            self.data = data
        return np.apply_along_axis(lambda x: lfilter(self.b, self.a, x), 0, self.data)


class SavgolFiltering(PreprocessingFiltering):
    def __init__(self, data=None):
        super().__init__(data)
        self.window_length, self.polyorder = 1001, 2

    def __name__(self):
        return 'SavgolFiltering'

    def __condition__(self, window_length=None, polyorder=None):
        if window_length:
            self.window_length = window_length
        if polyorder:
            self.polyorder = polyorder
        return f'{self.__name__}: window_length = {self.window_length}, polyorder = {self.polyorder}'

    def filtering(self, data=None):
        if data is not None:
            self.data = data
        return np.apply_along_axis(lambda x: savgol_filter(x, self.window_length, self.polyorder), 0, self.data)


class SmOLS(PreprocessingFiltering):
    def __init__(self, data=None):
        super().__init__(data)
        self.frac = 0.1

    def __name__(self):
        return 'SmOLS'

    def __condition__(self, frac=None):
        if frac:
            self.frac = frac
        return f'{self.__name__}: frac = {self.frac}'

    def filtering(self, data=None):
        if data is not None:
            self.data = data
        return np.apply_along_axis(lambda x: sm.nonparametric.lowess(x, np.arange(len(x)), frac=self.frac, it=0)[:, 1], 0, self.data)


class HampelFiltering(PreprocessingFiltering):
    def __init__(self, data=None, windows_size=100, n_sigmas=2, percentile=0.75, torch=False, device='cpu'):
        '''
        :param data: input data
        '''
        super().__init__(data)
        self.window_size = windows_size
        self.n_sigmas = n_sigmas
        self.percentile = percentile
        self.torch = torch
        self.device = device

    def __name__(self):
        return 'HampelFiltering'

    def __condition__(self, window_size=None, n_sigmas=None, percentile=None, torch=False, device='cpu'):
        if window_size:
            self.window_size = window_size
        if n_sigmas:
            self.n_sigmas = n_sigmas
        if percentile:
            self.percentile = percentile
        self.torch = torch
        self.device = device
        return f'{self.__name__}: window_size = {self.window_size}, n_sigmas = {self.n_sigmas}, percentile = {self.percentile}, torch = {self.torch}, device = {self.device}'

    def hampel(self, x, window_size, n_sigmas):
        """
        Hampel filter for outlier detection
        :param x: input data
        :param window_size: window size
        :param n_sigmas: number of sigmas
        :return: filtered data
        """
        k = 1/norm.ppf(self.percentile)  # scale factor
        if window_size % 2 == 0:
            window_size += 1
        x_pad = np.pad(x, (window_size//2, window_size//2), 'edge')
        medians = np.array([np.median(x_pad[i:i+window_size])
                            for i in range(len(x))])
        diff = np.abs(medians - x)
        med_abs_deviation = np.array(
            [np.median(diff[i:i+window_size]) for i in range(len(x))])
        threshold = n_sigmas * k * med_abs_deviation
        outliers = diff > threshold
        x[outliers] = medians[outliers]
        return x

    def tc_hampel(self, x, window_size, n_sigmas):
        """
        Hampel filter for outlier detection
        :param x: input data
        :param window_size: window size
        :param n_sigmas: number of sigmas
        :return: filtered data
        """
        k = 1/norm.ppf(self.percentile)
        if window_size % 2 == 0:
            window_size += 1
        x_pad = tc.nn.functional.pad(x.reshape(
            1, -1), (window_size//2, window_size//2), mode='replicate').reshape(-1)
        medians = tc.tensor([tc.median(x_pad[i:i+window_size])
                             for i in range(len(x))]).to(self.device)
        diff = tc.abs(medians - x).to(self.device)
        med_abs_deviation = tc.tensor(
            [tc.median(diff[i:i+window_size]) for i in range(len(x))]).to(self.device)
        threshold = n_sigmas * k * med_abs_deviation
        outliers = diff > threshold
        x[outliers] = medians[outliers]
        return x

    def filtering(self, data=None):
        if data is not None:
            self.data = data
        if self.device.type == 'mps':
            self.data = self.data.cpu().numpy()
            results = np.apply_along_axis(lambda x: self.hampel(
                x, self.window_size, self.n_sigmas), 0, self.data)
            results = tc.from_numpy(results).to(self.device)
            return results
        if self.torch:
            return torch_apply_along_axis(lambda x: self.tc_hampel(x, self.window_size, self.n_sigmas), 1, self.data)
        else:
            return np.apply_along_axis(lambda x: self.hampel(x, self.window_size, self.n_sigmas), 0, self.data)


class ButterBandpassFilter(PreprocessingFiltering):
    def __init__(self, data):
        super().__init__(data)
        self.fs, self.order, self.low, self.high = 2000, 5, 5, 600  # input
        nyq = 0.5 * self.fs
        self.low, self.high = self.low / nyq, self.high / nyq
        self.filter = self.butter_bandpass

    def __name__(self):
        return 'ButterBandpassFilter'

    def __condition__(self, low=None, high=None, fs=None, order=None, filter=None):
        if fs:
            self.fs = fs
        if order:
            self.order = order
        if filter:
            self.filter = filter
        nyq = 0.5 * self.fs  # whether fs is not given or not, self.fs will return last value, and nyq did not have to be remembered
        if low:
            self.low = low / nyq
        if high:
            self.high = high / nyq
        return f'{self.__name__}: low = {self.low}, high = {self.high}, fs = {self.fs}, order = {self.order}, filter = {self.filter}'

    def lfiltering(self, b, a):
        # type: ignore
        return np.apply_along_axis(lambda x: lfilter(b, a, x), 0, self.data)

    def butter_lowpass(self):
        b, a = butter(self.order, self.low, btype='low')
        return b, a

    def butter_bandpass(self):
        b, a = butter(self.order, [self.low, self.high], btype='band')
        return b, a

    def butter_bandpass_filter(self):
        b, a = self.filter()
        return self.lfiltering(b, a)
