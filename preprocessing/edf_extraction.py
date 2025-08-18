import datetime
import numpy as np
# import pyedflib
from pyedflib import EdfReader
from sklearn import preprocessing
import matplotlib

matplotlib.use('pdf')
from matplotlib import pyplot as plt
from scipy import signal
import sys

sys.path.append("/home/zengdifei/Documents/CHB_MIT/src")
import preprocessing.label_wrapper as label_wrapper
import utils.sharing_params as params
from pandas import DataFrame as df
import pywt
import math


class EdfFile(object):
    """
    Edf reader using pyedflib
    """

    def __init__(self, filename, patient_id=None):
        self._filename = filename
        self._patient_id = patient_id
        self._file = EdfReader(filename)

    def get_filename(self):
        return self._filename

    def get_n_channels(self):
        """
        Number of channels
        """
        return len(self._file.getSampleFrequencies())

    def get_n_data_points(self):
        """
        Number of data points
        """
        if len(self._file.getNSamples()) < 1:
            raise ValueError("Number of channels is less than 1")
        return self._file.getNSamples()[0]

    def get_channel_names(self):
        """
        Names of channels
        """
        return self._file.getSignalLabels()

    def get_channel_id(self):
        """
        Index of channels
        """
        channel_names = self.get_channel_names()
        channel_id = {}
        for ind, channel in enumerate(channel_names):
            if channel != "-":
                channel_id[channel] = ind
        return channel_id

    def get_channel_scalings(self):
        """
        Channel scalings as an array
        """
        out = np.zeros(self.get_n_channels())
        for i in range(self.get_n_channels()):
            out[i] = self._file.getPhysicalMaximum(i) - self._file.getPhysicalMinimum(i)
        return out

    def get_channel_absolute_mean(self):
        signals = self.get_data()
        out = np.mean(np.abs(signals), axis=0, keepdims=False)
        return out

    def get_channel_maximum_physical(self):
        """
        Channel physical maximum as an array
        """
        out = np.zeros(self.get_n_channels())
        for i in range(self.get_n_channels()):
            out[i] = self._file.getPhysicalMaximum(i)
        return out

    def get_channel_minimum_physical(self):
        """
        Channel physical minimum as an array
        """
        out = np.zeros(self.get_n_channels())
        for i in range(self.get_n_channels()):
            out[i] = self._file.getPhysicalMinimum(i)
        return out

    def get_channel_maximum_digital(self):
        """
        Channel physical maximum as an array
        """
        out = np.zeros(self.get_n_channels())
        for i in range(self.get_n_channels()):
            out[i] = self._file.getDigitalMaximum(i)
        return out

    def get_channel_minimum_digital(self):
        """
        Channel physical minimum as an array
        """
        out = np.zeros(self.get_n_channels())
        for i in range(self.get_n_channels()):
            out[i] = self._file.getDigitalMinimum(i)
        return out

    def get_file_duration(self):
        """
        Returns the file duration in seconds
        """
        return self._file.getFileDuration()

    def get_sampling_rate(self):
        """
        Get the frequency
        """
        if len(self._file.getSampleFrequencies()) < 1:
            raise ValueError("Number of channels is less than 1")
        return self._file.getSampleFrequency(0)

    def get_channel_data(self, channel_id):
        """
        Get raw data for a single channel
        """
        if channel_id >= self.get_n_channels() or channel_id < 0:
            raise ValueError("Illegal channel id selected %d" % channel_id)
        raw_data = self._file.readSignal(channel_id)
        clear_data = clear_raw_data(raw_data)
        return clear_data

    def get_standard_channel_data(self, channel_id):
        """
        Get z-normalized raw data for a signal channel
        """
        raw_data = self.get_channel_data(channel_id)
        raw_data = np.expand_dims(np.array(raw_data), axis=1)
        s = preprocessing.StandardScaler()
        params = s.fit(raw_data)
        standard_data = params.fit_transform(raw_data)
        return np.squeeze(standard_data), params.mean_, params.var_

    def get_minmax_channel_data(self, channel_id):
        """
        Get minmaxscalar raw data for a signal channel
        """
        raw_data = self.get_channel_data(channel_id)
        raw_data = np.expand_dims(np.array(raw_data), axis=1)
        s = preprocessing.MinMaxScaler()
        params = s.fit(raw_data)
        minmax_data = params.fit_transform(raw_data)
        return np.squeeze(minmax_data)

    def get_proposed_channel_data(self, channel_id):
        """
        Get proposed raw data for a signal channel
        """
        raw_data = self.get_channel_data(channel_id)
        raw_data = np.array(raw_data)
        proposed_data = raw_data / np.mean(np.abs(raw_data))
        return proposed_data

    def get_data(self):
        """
        Get raw data for all channels
        """
        output_data = np.zeros((self.get_n_data_points(), self.get_n_channels()))
        for i in range(self.get_n_channels()):
            output_data[:, i] = self.get_channel_data(i)
        return output_data

    def get_standard_data(self):
        """
        Get z-normalized raw data for all channels
        """
        output_data = np.zeros((self.get_n_data_points(), self.get_n_channels()))
        for i in range(self.get_n_channels()):
            output_data[:, i] = self.get_standard_channel_data(i)[0]
        return output_data

    def get_minmax_data(self):
        """
        Get minmax raw data for all channels
        """
        output_data = np.zeros((self.get_n_data_points(), self.get_n_channels()))
        for i in range(self.get_n_channels()):
            output_data[:, i] = self.get_minmax_channel_data(i)
        return output_data

    def get_proposed_data(self):
        """
        Get proposed raw data for all channels
        """
        output_data = np.zeros((self.get_n_data_points(), self.get_n_channels()))
        for i in range(self.get_n_channels()):
            output_data[:, i] = self.get_proposed_channel_data(i)
        return output_data

    def get_start_datetime(self):
        """
        Get the starting date and time
        """
        return self._file.getStartdatetime()

    def get_end_datetime(self):
        return self._file.getStartdatetime() + datetime.timedelta(seconds=self._file.getFileDuration())

    def display(self):
        # 采样率 256Hz
        print(self.get_sampling_rate())
        # 这份文件下的通道名
        print(self.get_channel_names())
        # 每个通道所对应的列号，便于提取数据
        print(len(self.get_channel_id()), self.get_channel_id())
        # 某channel_id通道下的数据
        data = np.array(self.get_channel_data(0))
        print(data)
        x = np.arange(0, data.shape[0]) / 256
        plt.plot(x, data, 'b')
        plt.savefig('/home/zengdifei/Output/CHB_MIT/test/raw_signal.png')
        plt.close()
        # 某channel_id通道下的standard数据
        data, mean, var = self.get_standard_channel_data(0)
        print(data, mean, var)
        x = np.arange(0, data.shape[0]) / 256
        plt.plot(x, data, 'b')
        plt.savefig('/home/zengdifei/Output/CHB_MIT/test/standard_signal.png')
        plt.close()
        # 某channel_id通道下的minmax数据
        data = self.get_minmax_channel_data(0)
        print(data)
        x = np.arange(0, data.shape[0]) / 256
        plt.plot(x, data, 'b')
        plt.savefig('/home/zengdifei/Output/CHB_MIT/test/minmax_signal.png')
        plt.close()
        # 某channel_id通道下的proposed数据
        data = self.get_proposed_channel_data(0)
        print(data)
        x = np.arange(0, data.shape[0]) / 256
        plt.plot(x, data, 'b')
        plt.savefig('/home/zengdifei/Output/CHB_MIT/test/proposed_signal.png')
        plt.close()
        # 每个通道下的scale, max, min from physical or digital
        print(self.get_channel_scalings())
        print(self.get_channel_maximum_digital())
        print(self.get_channel_maximum_physical())
        print(self.get_channel_minimum_digital())
        print(self.get_channel_minimum_physical())
        # 每个通道下的绝对平均值
        print(self.get_channel_absolute_mean())
        # 所有通道下的数据
        print(self.get_data())
        # 这份文件的开始时间
        print(self.get_start_datetime())
        # 这份文件的结束时间
        print(self.get_end_datetime())
        # 这份文件的持续时间（秒）
        print(self.get_file_duration())
        # 这份文件中有多少个通道
        print(self.get_n_channels())
        # 这份文件中共有多少个采样点 921600 = 60 * 60 * 256
        print(self.get_n_data_points())
        # 文件名
        print(self.get_filename())


def clear_raw_data(raw_signal):
    """
    filter signal
    :param raw_signal: [signal]
    :return: [signal]
    """
    # 滤除工频60Hz附近的频率
    filter_signal = butter_bandstop_filter(raw_signal, 57, 63, 256)
    # 滤除工频120Hz附近的频率
    filter_signal = butter_bandstop_filter(filter_signal, 117, 123, 256)
    # 滤除小于1Hz的频率
    filter_signal = butter_highpass_filter(filter_signal, 1, 256)
    # 滤除高于70Hz的频率
    # filter_signal = butter_lowpass_filter(filter_signal, 70, 256)
    return filter_signal


def butter_bandstop_filter(data, lowcut, highcut, fs, order=6):
    """
    带阻滤波器
    :param data: [signal]
    :param lowcut: low cut frequency
    :param highcut: high cut frequency
    :param fs: sample_rate
    :param order: order
    :return: [signal]
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=6):
    """
    高通滤波器
    :param data: [signal]
    :param cutoff: cutoff frequency
    :param fs: sample_rate
    :param order: order
    :return: [signal]
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=6):
    """
    低通滤波器
    :param data: [signal]
    :param cutoff: cutoff frequency
    :param fs: sample_rate
    :param order: order
    :return: [signal]
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = signal.lfilter(b, a, data)
    return y


def T_mean(input):
    """
    均值
    :param input: [signal]
    :return: mean
    """
    return np.array(input).mean()


def T_var(input):
    """
    方差
    :param input: [signal]
    :return: var
    """
    return np.array(input).var()


def T_abs_mean(input):
    """
    整流平均值
    :param input: [signal]
    :return: abs_mean
    """
    arr = np.array(input)
    arr = np.abs(arr)
    return arr.mean()


def T_std(input):
    """
    标准差
    :param input: [signal]
    :return: std
    """
    return np.array(input).std()


def T_ptp(input):
    """
    峰峰值
    :param input: [signal]
    :return: ptp
    """
    return np.array(input).ptp()


def T_skewness(input):
    """
    偏度
    :param input: [signal]
    :return: skewness
    """
    arr = df(input)
    return arr.skew()


def T_kurtosis(input):
    """
    峰度
    :param input: [signal]
    :return: kurtosis
    """
    arr = df(input)
    return arr.kurt()


def T_rms(input):
    """
    均方根
    :param input: [signal]
    :return: rms
    """
    return math.sqrt(np.power(np.array(input), 2).mean())


def T_Xr(input):
    """
    方根幅值
    :param input: [signal]
    :return: Xr
    """
    return math.pow(np.sqrt(np.abs(np.array(input))).mean(), 2)


def T_arv(input):
    """
    整流平均值
    :param input: [signal]
    :return: arv
    """
    return np.abs(np.array(input)).mean()


def T_S(input):
    """
    波形因子
    :param input: [signal]
    :return: S
    """
    return T_rms(input) / T_arv(input)


def T_C(input):
    """
    峰值因子
    :param input: [signal]
    :return: C
    """
    return np.array(input).max() / T_rms(input)


def T_I(input):
    """
    脉冲因子
    :param input: [signal]
    :return: I
    """
    return T_ptp(input) / T_abs_mean(input)


def T_L(input):
    """
    裕度因子
    :param input: [signal]
    :return: L
    """
    return T_ptp(input) / T_Xr(input)


def T_zero_crossing(input):
    """
    衡量信号穿过0点的频率，计算穿过0点的次数除以采样的总点数
    :param input: [signal]
    :return: ratio
    """
    tot = 0
    for i in range(1, len(input)):
        if (input[i] < 0 and input[i - 1] > 0) or (input[i] > 0 and input[i - 1] < 0):
            tot += 1
    return tot / len(input)


def F_STFT(input, rate):
    """
    短时傅里叶变换
    :param input: [signal]
    :param rate: sample_rate
    :return: [rate // 2, spectrum]
    """
    n = len(input)
    epoch = n // rate
    fft = np.fft.fft(input)[range(n // 2)] / (n // 2)
    fft = np.reshape(fft, [-1, epoch])
    fft = np.mean(fft, axis=1)
    return fft


def F_PSD(input, rate):
    """
    求信号的PSD
    :param input: [signal]
    :param rate: sample_rate
    :return: [rate // 2, energy]
    """
    [_, _, spec_Sxx] = signal.spectrogram(input, fs=rate, return_onesided=True, scaling='density')
    spec_Sxx = np.mean(spec_Sxx, axis=1)[range(rate // 2)]
    return spec_Sxx


def F_DWT(input):
    """
    求信号的db4离散小波变换
    :param input: [signal]
    :return: coeffs
    """
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(input, db4, level=7)
    return coeffs


def F_CWT(input, rate):
    """
    求信号的cgau8连续小波变换
    :param input: [signal, rate]
    :return: [cwtmatr, frequency]
    """
    wavename = 'cgau8'
    totalscal = 128
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(input, scales, wavename, 1.0 / rate)
    return cwtmatr, frequencies


def F_band_mean(input, frequency):
    """
    求每个频带内的平均能量
    :param input: [frequency_psd]
    :return: [band_psd]
    """
    if input.shape[0] != frequency.shape[0]:
        raise Exception('The frequency size is different in input and discription!')
    mean_psd = [[], [], [], [], [], []]
    for index in range(input.shape[0]):
        for fqc_band in params.band.keys():
            fqc = frequency[index]
            if fqc >= params.band[fqc_band][0] and fqc < params.band[fqc_band][1]:
                mean_psd[params.band[fqc_band][2]].append(input[index])
    for band in range(6):
        mean_psd[band] = np.stack(mean_psd[band], axis=0)
        mean_psd[band] = np.mean(mean_psd[band], axis=0)
    return np.stack(mean_psd, axis=0)


def check_file_channel(edf_file):
    """
    检查文件的信号通道是否包含normal_signal
    :param edf_file: edf_file
    :return: True / False
    """
    channels = edf_file.get_channel_names()
    for channel in params.normal_signal:
        if not channel in channels:
            return False
    return True


def max_abs_coefficient(input):
    """
    求i通道对其余j通道的最大绝对相关性
    :param input: [signal, time]
    :return: [signal, max_abs_coeff]
    """
    arr = np.array(input)
    coeff = np.corrcoef(arr)
    abs_coeff = np.abs(coeff)
    max_abs_coeff = np.max(abs_coeff, axis=1)
    return max_abs_coeff


def test_fft():
    """
    测试fft实现是否正确
    """
    Fs = 256
    Ts = 1.0 / Fs
    t = np.arange(0, 2, Ts)

    y = np.sin(2 * np.pi * 25 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 75 * t)
    # y = butter_lowpass_filter(y, 26, Fs)
    # y = butter_highpass_filter(y, 74, Fs)
    # y = butter_bandstop_filter(y, 26, 74, Fs)
    n = len(y)
    k = np.arange(n)
    T = n / Fs
    frq = k / T
    frq1 = frq[range(int(n / 2))]

    YY = np.fft.fft(y)
    Y = np.fft.fft(y) / (n / 2)
    Y1 = Y[range(int(n / 2))]

    [spec_f, _, spec_Sxx] = signal.spectrogram(y, fs=256, return_onesided=True, scaling='density')
    spec_Sxx = np.mean(spec_Sxx, axis=1)

    fig, ax = plt.subplots(5, 1)

    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(frq, abs(YY), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

    ax[2].plot(frq, abs(Y), 'G')  # plotting the spectrum
    ax[2].set_xlabel('Freq (Hz)')
    ax[2].set_ylabel('|Y(freq)|')

    ax[3].plot(frq1, abs(Y1), 'B')  # plotting the spectrum
    ax[3].set_xlabel('Freq (Hz)')
    ax[3].set_ylabel('|Y(freq)|')

    ax[4].plot(spec_f, spec_Sxx, 'y')
    ax[4].set_xlabel('Freq (Hz)')
    ax[4].set_ylabel('|Y(freq)|')

    print(spec_Sxx[25], spec_Sxx[50], spec_Sxx[75])

    plt.savefig("/home/zengdifei/Output/CHB_MIT/test/fft.png")
    plt.close()


def test_wavelet():
    sampling_rate = 256
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    f1 = 100
    f2 = 50
    f3 = 25
    data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                        [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
                         lambda t: np.sin(2 * np.pi * f3 * t)])
    wavename = 'cgau8'
    totalscal = 128
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    print(cwtmatr.shape)
    print(frequencies.shape)
    print(frequencies)
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"300Hz 200Hz 100Hz Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('/home/zengdifei/Output/CHB_MIT/test/wavelet_test.png')
    plt.close()


def feature_extraction():
    """
    analysis each kind of feature extracted by raw signal
    """
    for patient in params.patients:
        filename = params.dataset_dir + "/chb" + patient + "/chb" + patient + "-summary.txt"
        patient_label_wrapper = label_wrapper.LabelWrapper(filename)
        seizure_list, file_list = patient_label_wrapper.get_seizure_list()
        seizure_list = [seizure[0] for seizure in seizure_list]
        # time_domain
        # abs_mean_list = []
        # std_list = []
        for file in file_list:
            dir = params.dataset_dir + file[0:5] + "/" + file
            edf_file = EdfFile(dir)
            if file in seizure_list:
                save_file = file + "_seizure"
            else:
                save_file = file
            if check_file_channel(edf_file):
                # time_domain
                # if file not in seizure_list:
                #     data = np.array(edf_file.get_channel_data(0))
                #     abs_mean = np.abs(data).mean()
                #     std = data.std()
                #     abs_mean_list.append(abs_mean)
                #     std_list.append(std)
                #     print(file, abs_mean, std)

                # proposed_signal
                # data = edf_file.get_proposed_channel_data(0)
                # x = np.arange(0, data.shape[0]) / 256
                # plt.plot(x, data, 'b')
                # plt.savefig('/home/zengdifei/Output/CHB_MIT/test/proposed_signal/%s.png' % save_file)
                # plt.close()

                # zero_crossing
                # segment_length = 10
                # data = edf_file.get_channel_data(0)
                # rate = edf_file.get_sampling_rate()
                # duration = edf_file.get_file_duration()
                # zero_crossing_data = np.zeros([duration - (segment_length - 1)])
                # for i in range(duration - (segment_length - 1)):
                #     zero_crossing_data[i] = T_zero_crossing(data[i * rate:(i + segment_length) * rate])
                # x = np.arange(0, duration - (segment_length - 1))
                # plt.plot(x, zero_crossing_data, 'b')
                # plt.savefig('/home/zengdifei/Output/CHB_MIT/test/zero_crossing/%s.png' % save_file)
                # plt.close()

                # PSD_map
                # import seaborn as sns
                #
                # data = edf_file.get_proposed_channel_data(0)
                # rate = edf_file.get_sampling_rate()
                # duration = edf_file.get_file_duration()
                # data = data[0 * rate:duration * rate]
                # psd = []
                # for st in range(0, duration - params.epoch_length + 1, params.epoch_length // 2):
                #     psd.append(T_PSD(data[st * rate: (st + params.epoch_length) * rate], rate))
                # psd = np.transpose(np.array(psd), [1, 0])
                # mean_psd = band_mean(psd)
                # sns.heatmap(mean_psd)
                # plt.savefig('/home/zengdifei/Output/CHB_MIT/test/PSD_map/%s.png' % save_file)
                # plt.close()
                pass
            del edf_file

        # time_domain
        # abs_mean_arr = np.array(abs_mean_list)
        # x = np.arange(0, abs_mean_arr.shape[0], 1)
        # plt.plot(x, abs_mean_arr, 'b')
        # plt.savefig(params.dataset_dir + "/chb" + patient + "/chb" + patient + "_0_abs_mean.png")
        # plt.close()
        # std_arr = np.array(std_list)
        # x = np.arange(0, std_arr.shape[0], 1)
        # plt.plot(x, std_arr, 'b')
        # plt.savefig(params.dataset_dir + "/chb" + patient + "/chb" + patient + "_0_std.png")
        # plt.close()
        break


if __name__ == "__main__":
    # filename = "/home/zengdifei/Dataset/CHB_MIT/chb19/chb19_01.edf"
    # edf_file = EdfFile(filename)
    # edf_file.display()
    # feature_extraction()
    # test_fft()
    test_wavelet()
