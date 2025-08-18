import numpy as np
import sys

sys.path.append("/home/zengdifei/Documents/CHB_MIT/src")
import preprocessing.edf_extraction as edf_extraction
import matplotlib

matplotlib.use('pdf')
from matplotlib import pyplot as plt
import utils.sharing_params as params


def draw():
    file_dir = "/home/zengdifei/Dataset/CHB_MIT/chb18/chb18_35.edf"
    edf_file = edf_extraction.EdfFile(file_dir)
    signal_dict = edf_file.get_channel_id()
    sampling_rate = edf_file.get_sampling_rate()
    st = 127
    en = 148
    standard_out = np.zeros((len(params.normal_signal), (en - st) * sampling_rate), dtype=np.float64)
    for i, channel in enumerate(params.normal_signal):
        try:
            id = signal_dict[channel]
            standard_channel_data, _, _ = edf_file.get_standard_channel_data(id)
            standard_out[i] = standard_channel_data[st * sampling_rate:en * sampling_rate]
        except Exception as e:
            print('Failed to extract channel %s in file %s with exception %s' % (channel, file_dir, e))
    input = standard_out
    rate = sampling_rate
    t = input.shape[1] / rate
    time = np.arange(0, t, 1.0 / rate)
    ratio = 15
    signals = len(params.normal_signal)
    for i in range(signals):
        plt.plot(time, input[signals - i - 1] + i * ratio, 'b')
    reverse_signal = params.normal_signal
    reverse_signal.reverse()
    plt.yticks(np.arange(0, signals, 1) * ratio, reverse_signal)
    plt.xlabel("seconds")
    plt.ylabel("channel")
    plt.savefig("/home/zengdifei/Output/CHB_MIT/test/innovation_raw.png")
    plt.close()

    correlation_matrix = np.corrcoef(input)
    sub_correlation_matrix = np.zeros((4, 4), dtype=np.float64)
    """
    sub_correlation_matrix = [1.000, 0.786, 0.816, 0.722,
                              0.786, 1.000, 0.913, 0.757,
                              0.816, 0.913, 1.000, 0.805,
                              0.722, 0.757, 0.805, 1.000]
    """
    index = [0, 4, 10, 14]
    for i in range(4):
        for j in range(4):
            sub_correlation_matrix[i][j] = correlation_matrix[index[i]][index[j]]
    print(sub_correlation_matrix)

    print(sub_correlation_matrix.mean(), correlation_matrix.mean())
    """
    0.8500 0.2625
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hot_img = ax.matshow(np.abs(correlation_matrix), vmin=0, vmax=1)
    fig.colorbar(hot_img)
    ticks = np.arange(0, len(params.normal_signal), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(params.normal_signal, rotation=90)
    ax.set_yticklabels(params.normal_signal)
    plt.savefig("/home/zengdifei/Output/CHB_MIT/test/innovation_correlation.png")
    plt.close()


if __name__ == "__main__":
    draw()
