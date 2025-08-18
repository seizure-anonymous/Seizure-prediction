import numpy as np
import sys
import torch
sys.path.append("/home/zhongying/CHB_MIT/src")
import preprocessing.edf_extraction as edf_extraction
import matplotlib

matplotlib.use('pdf')
from matplotlib import pyplot as plt
import math
import os
import random
from datetime import timedelta
import utils.sharing_params as params
from tqdm import tqdm
from scipy import signal

import time
from multiprocessing import Pool

# import seaborn as sns

class ExtractSignal(object):
    def __init__(self, signal):
        self._signal = signal
        self._dataset_dir = params.dataset_dir
        self.edf_file = None
        self.Standard_channel_data_list = {}

    def file2dir(self, file):
        return self._dataset_dir + file[0:5] + "/" + file

    def get_signal(self, sample):
        file = sample[0]
        st = sample[1]
        en = sample[2]
        file_dir = self.file2dir(file)
        if self.edf_file==None or sample[0]!=self.edf_file.get_filename().split('/')[-1]:
            self.edf_file = edf_extraction.EdfFile(file_dir)
            signal_dict = self.edf_file.get_channel_id()
            self.Standard_channel_data_list={}
            # for i in range (len(signal_dict)):
            #     temp=[]
            #     self.Standard_channel_data_list.append(temp)
            for i, channel in enumerate(signal_dict):
                try:
                    id = signal_dict[channel]
                    Standard_channel_data, _, _ = self.edf_file.get_standard_channel_data(id)
                    self.Standard_channel_data_list[channel]=(Standard_channel_data)
                except Exception as e:
                    print('Failed to extract channel %s in file %s with exception %s' % (channel, file, e))
        # signal_dict = self.edf_file.get_channel_id()
        sampling_rate = self.edf_file.get_sampling_rate()
        # Raw_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        Standard_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        # MinMax_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        # MeanBase_out = np.zeros((len(self._signal), (en - st) * sampling_rate), dtype=np.float64)
        for i, channel in enumerate(self._signal):
            try:
                # id = signal_dict[channel]
                # Standard_channel_data, _, _ = self.edf_file.get_standard_channel_data(id)
                Standard_out[i] = self.Standard_channel_data_list[channel][st * sampling_rate:en * sampling_rate]
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



    def all_channel_t_PSD(self, input, rate):
        def t2_PSD(input):
            freqs, psd = signal.welch(input, rate, nperseg=4*rate)
            # psd = np.abs(edf_extraction.F_PSD(input, rate))
            theta=alpha=beta=gama1=gama2=gama3=gama4=gama5=0.0
            for index, freq in enumerate(freqs):
                if 4<=freq<8:
                    theta=theta+psd[index]
                elif 8<=freq<13:
                    alpha=alpha+psd[index]
                elif 13<=freq<30:
                    beta=beta+psd[index]
                elif 30<=freq<50:
                    gama1=gama1+psd[index]
                elif 50<=freq<70:
                    gama2=gama2+psd[index]
                elif 70<=freq<90:
                    gama3=gama3+psd[index]
                elif 90<=freq<110:
                    gama4=gama4+psd[index]
                elif 110<=freq<128:
                    gama5=gama5+psd[index]
            return theta,alpha,beta,gama1,gama2,gama3,gama4,gama5

        feature_out = np.zeros((input.shape[0], 8+8+56//2))
        for channel in range(input.shape[0]):
            # theta, alpha, beta, gama1, gama2, gama3, gama4, gama5
            abs_band= (np.array(t2_PSD(input[channel])))
            abs_band=np.expand_dims(abs_band, 1)
            relative_band=abs_band/abs_band.sum()
            abs_band = np.log(abs_band)
            relative_band = np.log(relative_band)
            band_ratio = abs_band-abs_band.T

            # abs_band_inverse_T=(1.0/abs_band).T
            # band_ratio=np.dot(abs_band,abs_band_inverse_T)
            band_ratio_feature=np.array([])
            for row in range (len(band_ratio)):
                band_ratio_feature=np.append(band_ratio_feature, band_ratio[row][row+1:])


            feature_one_channel=np.concatenate((abs_band.squeeze(1), relative_band.squeeze(1), band_ratio_feature))
            feature_out[channel] = feature_one_channel
        return feature_out

# class run_extract_capsule:
#     def __init__(self):
#         self.edf_file=None
def run_extract(sample, label, visual, extract):

    # Raw_signal, Standard_signal, MinMax_signal, MeanBase_signal, signal_sampling_rate = extract.get_signal(sample)

    Standard_signal, signal_sampling_rate= extract.get_signal(sample)
    if signal_sampling_rate == 0:
        # return False, None, None, None, None
        return False, None
    else:
        # Raw_feature_signal = extract.all_channel_t2d(Raw_signal, signal_sampling_rate)
        Standard_feature_signal = extract.all_channel_t_PSD(Standard_signal, signal_sampling_rate)
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
        os.makedirs(dir)
    return dir

def build_tfrecords_pred(path, file_name):
    print("%s %s start: %s"%(path, file_name, time.time()))
    nums_ratio = params.seizure_ratio[params.input_length]
    label=file_name[0:-4]
    file=os.path.join(path, file_name)
    f = open(file, 'r')
    extractions = []
    for line in f:
        tmp = line.strip().split(" ")
        extractions.append((tmp[0], int(tmp[1]), int(tmp[2])))
    f.close()
    real_samples = 0
    data_lable = []
    data_feature = []
    # data={}
    extractor = ExtractSignal(params.normal_signal)
    for extraction in tqdm(extractions, desc=file):
        success, Standard_fft = run_extract(extraction, label, False, extractor)
        if success:
            data_lable.append(params.label_dict_pred[label])
            data_feature.append(Standard_fft)
            real_samples += 1
        else:
            print("Extraction failed: %s" %str(extraction))
    print("%s %s end: %s"%(path, file_name, time.time()))
    torch.save(data_lable, f=check_dir(path + "/%s/" % label) + 'label')
    torch.save(data_feature, f=check_dir(path + "/%s/" % label) + 'feature')

if __name__ == '__main__':
    split_dir=os.path.join('/data/zhengruifeng/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/all24_preictal_1h_post_pre_ictal_1h_len_2/', 'pred_v2','split')
    # split_dir='/data/zhengruifeng/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/77_preictal_1h_post_pre_ictal_4h_len_2/pred_v2/split/02/2/'
    file_list=[]
    root_list=[]
    for root, ds, fs in os.walk(split_dir):
        for file in fs:
            build_tfrecords_pred(path=root, file_name=file)
    #         file_list.append(file)
    #         root_list.append(root)
    # pool
    # pool.map(build_tfrecords_pred, root_list, file_list)
    # pool.close()
    # pool.join()
    # build_tfrecords_pred(path=root, file_name=file)

    # build_tfrecords('01')
