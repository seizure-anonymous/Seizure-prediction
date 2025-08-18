import numpy as np
import tensorflow as tf
import sys

# sys.path.append("/home/zengdifei/Documents/CHB_MIT/src")
import preprocessing.edf_extraction as edf_extraction
import matplotlib

matplotlib.use('pdf')
from matplotlib import pyplot as plt
import math
import os
import random
from datetime import timedelta
import utils.sharing_params as params


class ExtractSignal(object):
    def __init__(self, signal):
        self._signal = signal
        self._dataset_dir = params.dataset_dir

    def file2dir(self, file):
        return self._dataset_dir + file[0:5] + "/" + file

    def get_signal(self, sample):
        file = sample[0]
        st = sample[1]
        en = sample[2]
        file_dir = self.file2dir(file)
        edf_file = edf_extraction.EdfFile(file_dir)
        signal_dict = edf_file.get_channel_id()
        sampling_rate = edf_file.get_sampling_rate()
        # Raw_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        Standard_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        # MinMax_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        # MeanBase_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        for i, channel in enumerate(self._signal):
            try:
                id = signal_dict[channel]
                # Raw_channel_data = edf_file.get_channel_data(id)
                Standard_channel_data, _, _ = edf_file.get_standard_channel_data(id)
                # MinMax_channel_data = edf_file.get_minmax_channel_data(id)
                # MeanBase_channel_data = edf_file.get_proposed_channel_data(id)
                # Raw_out[i] = Raw_channel_data[st * sampling_rate:en * sampling_rate]
                Standard_out[i] = Standard_channel_data[st * sampling_rate:en * sampling_rate]
                # MinMax_out[i] = MinMax_channel_data[st * sampling_rate:en * sampling_rate]
                # MeanBase_out[i] = MeanBase_channel_data[st * sampling_rate:en * sampling_rate]
            except Exception as e:
                print('Failed to extract channel %s in file %s with exception %s' % (channel, file, e))
                # return Raw_out, Standard_out, MinMax_out, MeanBase_out, 0
                return Standard_out, 0
        # return Raw_out, Standard_out, MinMax_out, MeanBase_out, sampling_rate
        return Standard_out, sampling_rate

    def dRaw_Raw_signal(self, sample, input, rate, label):
        t = input.shape[1] / rate
        time = np.arange(0, t, 1.0 / rate)
        ratio = 15
        for i in range(input.shape[0]):
            plt.plot(time, input[i] + i * ratio, 'b')
        plt.yticks(np.arange(0, input.shape[0], 1) * ratio, self._signal)
        plt.plot(time, input[0], 'b')
        plt.xlabel("seconds")
        plt.ylabel("channel")
        plt.title(label)
        # st = edf_extraction.EdfFile(self.file2dir(sample[0])).get_start_datetime()
        # sample_st = (st + timedelta(seconds=sample[1])).strftime("%H:%M:%S")
        # sample_en = (st + timedelta(seconds=)).strftime("%H:%M:%S")
        # plt.savefig(
        #     "/home/zengdifei/Output/CHB_MIT/test/raw_eeg/%s_%s_%s_%s.png" % (
        #     label, sample[0].split('.')[0], sample_st, sample_en))
        plt.savefig(
            "/home/zhengruifeng/Output/CHB_MIT/test/raw_eeg/%s_%s_%d_%d.png" % (
            label, sample[0].split('.')[0], sample[1], sample[2]))
        plt.close()

    def dRaw_fft_signal(self, sample, input, rate, label, ymin, ymax, name):
        fs = rate
        time = np.arange(0, fs // 2, 1)
        time = np.delete(time, np.s_[117: 123 + 1], axis=0)
        time = np.delete(time, np.s_[57: 63 + 1], axis=0)
        time = np.delete(time, np.s_[0: 1 + 1], axis=0)
        for i in range(input.shape[0]):
            plt.plot(time, input[i] + i * (ymax - ymin), 'b')
        plt.yticks(range(0, input.shape[0] * (ymax - ymin), (ymax - ymin)), self._signal[0:input.shape[0]])
        plt.plot(time, input[0], 'b')
        plt.axis([0, 128, ymin, ymax])
        plt.xlabel("frequency")
        plt.ylabel("channel")
        plt.title("fft_" + label)
        st = edf_extraction.EdfFile(self.file2dir(sample[0])).get_start_datetime()
        sample_st = (st + timedelta(seconds=sample[1])).strftime("%H:%M:%S")
        sample_en = (st + timedelta(seconds=sample[2])).strftime("%H:%M:%S")
        plt.savefig(
            "/home/zhengruifeng/Output/CHB_MIT/test/fft_eeg/%s_%s_%s_%s_%s.png" % (
                name, label, sample[0].split('.')[0], sample_st, sample_en))
        plt.close()

    def all_channel_t2d(self, input, rate):
        def t2d(input):
            feature = np.zeros((params.epoches, 275))
            cwtmatr, frequencies = edf_extraction.F_CWT(input, rate)
            for t in range(params.epoches):
                st = t * rate
                en = (t + params.epoch_length) * rate
                matr = np.abs(np.mean(cwtmatr[:, st:en], axis=1, keepdims=False))
                signal = input[st:en]
                # fft = edf_extraction.F_STFT(signal, rate)
                psd = np.abs(edf_extraction.F_PSD(signal, rate))
                feature[t][0:128] = np.array(psd, dtype=np.float64)
                feature[t][128:134] = np.array(edf_extraction.F_band_mean(psd, np.arange(0, 128, 1)), dtype=np.float64)
                feature[t][134:261] = np.array(matr, dtype=np.float64)
                feature[t][261:267] = np.array(edf_extraction.F_band_mean(matr, frequencies), dtype=np.float64)
                feature[t][268] = np.array(edf_extraction.T_mean(signal), dtype=np.float64)
                feature[t][269] = np.array(edf_extraction.T_abs_mean(signal), dtype=np.float64)
                feature[t][270] = np.array(edf_extraction.T_kurtosis(signal), dtype=np.float64)
                feature[t][271] = np.array(edf_extraction.T_ptp(signal), dtype=np.float64)
                feature[t][272] = np.array(edf_extraction.T_skewness(signal), dtype=np.float64)
                feature[t][273] = np.array(edf_extraction.T_zero_crossing(signal), dtype=np.float64)
                feature[t][274] = np.array(edf_extraction.T_std(signal), dtype=np.float64)

            return feature

        feature_out = np.zeros((input.shape[0], params.epoches, 275))
        for channel in range(input.shape[0]):
            feature = t2d(input[channel])
            feature_out[channel] = feature
        return feature_out


def run_extract(signal, sample, label, visual):
    extract = ExtractSignal(signal)
    # Raw_signal, Standard_signal, MinMax_signal, MeanBase_signal, signal_sampling_rate = extract.get_signal(sample)
    Standard_signal, signal_sampling_rate = extract.get_signal(sample)
    if signal_sampling_rate == 0:
        # return False, None, None, None, None
        return False, None
    else:
        # Raw_feature_signal = extract.all_channel_t2d(Raw_signal, signal_sampling_rate)
        Standard_feature_signal = extract.all_channel_t2d(Standard_signal, signal_sampling_rate)
        # MinMax_feature_signal = extract.all_channel_t2d(MinMax_signal, signal_sampling_rate)
        # MeanBase_feature_signal = extract.all_channel_t2d(MeanBase_signal, signal_sampling_rate)
        if visual:
            extract.dRaw_Raw_signal(sample, Standard_signal, signal_sampling_rate, label)
            # print(Raw_feature_signal.shape)
            pass
        # return True, Raw_feature_signal, Standard_feature_signal, MinMax_feature_signal, MeanBase_feature_signal
        return True, Standard_feature_signal


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def make_tfrecords(label, patient_specific=None, label_num=None):
    """
    :param label: 要生成数据集的类别
    :param patient_specific: 如果是None的话则代表生成cross-validation实验的数据集，
                             否则是一个"02%d"代表某个病人生成patient_specific实验的数据集
    :return:
    """
    if patient_specific is None:
        father_dir = check_dir(params.tfRecords_dir + "/%s" % label)
        file = params.tfRecords_dir + "/split/" + label + ".txt"
    else:
        root_dir = check_dir(params.tfRecords_dir + "/patient_specific")
        grand_father_dir = check_dir(root_dir + "/chb" + patient_specific)
        father_dir = check_dir(grand_father_dir + "/%s" % label)
        file = params.tfRecords_dir + "/patient_specific/split/" + label + ".txt"
    f = open(file, 'r')
    extractions = []
    for line in f:
        tmp = line.strip().split(" ")
        if patient_specific is not None and not tmp[0].startswith("chb" + patient_specific):
            continue
        extractions.append((tmp[0], int(tmp[1]), int(tmp[2])))
    f.close()
    if label_num is None:
        samples_all = len(extractions)
    else:
        samples_all = min(label_num, len(extractions))
    samples_in_per_tfrecord = math.floor(samples_all / 10)
    real_samples = 0
    for file_num in range(10):
        writer = tf.python_io.TFRecordWriter(father_dir + "/%02d.tfrecords" % file_num)
        for index in range(int(samples_in_per_tfrecord)):
            if file_num * samples_in_per_tfrecord + index < samples_all:
                extraction = extractions[int(file_num * samples_in_per_tfrecord + index)]
                # success, Raw_fft, Standard_fft, MinMax_fft, MeanBase_fft = run_extract(params.normal_signal, extraction,
                #                                                                        label, False)
                success, Standard_fft = run_extract(params.normal_signal, extraction, label, False)
                if success:
                    tf_dict = {"label": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[np.array([params.label_dict[label]], dtype=np.int64).tobytes()]))}
                    for ind, channel in enumerate(params.normal_signal):
                        # tf_dict['Raw_' + channel] = tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[Raw_fft[ind].tobytes()]))
                        tf_dict['Standard_' + channel] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[Standard_fft[ind].tobytes()]))
                        # tf_dict['MinMax_' + channel] = tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[MinMax_fft[ind].tobytes()]))
                        # tf_dict['MeanBase_' + channel] = tf.train.Feature(
                        #     bytes_list=tf.train.BytesList(value=[MeanBase_fft[ind].tobytes()]))
                    sample = tf.train.Example(features=tf.train.Features(feature=tf_dict))
                    writer.write(sample.SerializeToString())
                    if patient_specific is None:
                        print("finish %d %s tfrecord-samples %.2f%%" % (
                            samples_in_per_tfrecord, label, 100 * ((index + 1) / samples_all + 0.1 * file_num)))
                    else:
                        print("chb%s finish %d %s tfrecord-samples %.2f%%" % (
                            patient_specific, samples_in_per_tfrecord, label,
                            100 * ((index + 1) / samples_all + 0.1 * file_num)))
                    real_samples += 1
        writer.close()
    return real_samples


def test_draw(sample_size):
    def sample(label, num):
        file = "/home/zhengruifeng/Dataset/CHB_MIT/tfRecords%d-%d/split/" % (
            params.epoch_length, params.input_length) + label + ".txt"
        f = open(file, 'r')
        extractions = []
        for line in f:
            tmp = line.strip().split(" ")
            extractions.append((tmp[0], int(tmp[1]), int(tmp[2])))
        f.close()
        sample_extractions = random.sample(extractions, num)
        for sample in sample_extractions:
            run_extract(params.normal_signal, sample, label, True)

    # fft_path = '/home/zengdifei/Output/CHB_MIT/test/fft_eeg'
    # os.mkdir(fft_path)
    raw_path = '/home/zhengruifeng/Output/CHB_MIT/test/raw_eeg'
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
    # sample("interictal", sample_size)
    sample("preictal-I", sample_size)
    sample("preictal-II", sample_size)
    sample("preictal-III", sample_size)
    sample("ictal", sample_size)


def build_tfrecords(patient_specific=None):
    nums_ratio = params.seizure_ratio[params.input_length]
    tfrecord_num = {}
    tfrecord_num["ictal"] = make_tfrecords("ictal", patient_specific)
    if patient_specific is not None:
        nums_others = nums_ratio * tfrecord_num["ictal"] * 10
    else:
        nums_others = nums_ratio * tfrecord_num["ictal"]
    tfrecord_num["interictal"] = make_tfrecords("interictal", patient_specific, nums_others)
    tfrecord_num["preictal-I"] = make_tfrecords("preictal-I", patient_specific, nums_others)
    tfrecord_num["preictal-II"] = make_tfrecords("preictal-II", patient_specific, nums_others)
    tfrecord_num["preictal-III"] = make_tfrecords("preictal-III", patient_specific, nums_others)
    if patient_specific is not None:
        f = open(params.tfRecords_dir + "/patient_specific/tfRecordsNums.txt", 'a')
        for key in tfrecord_num.keys():
            print("chb" + patient_specific, key, tfrecord_num[key], file=f)
        print("------------------------", file=f)
        f.close()
    else:
        for key in tfrecord_num.keys():
            print(key, tfrecord_num[key])


if __name__ == '__main__':
    # patient-specific
    # for patient_specific in ['01','02','03']:
    for patient_specific in ['04', '05', '06', '07']:
    # for patient_specific in ['08', '09', '10', '11']:
    # for patient_specific in ['12', '13', '14', '15']:
    # for patient_specific in ['16', '17', '18', '19']:
    # for patient_specific in ['20', '21', '22', '23']:
    # for patient_specific in ['06']:
        print(patient_specific)
        build_tfrecords(patient_specific)
    # cross-validation
    # build_tfrecords()
    # test_draw(200)
