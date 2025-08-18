import tensorflow as tf
import numpy as np
import sys

sys.path.append("/home/zengdifei/Documents/CHB_MIT/src")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import time
import utils.sharing_params as params


class ReadTFRecords(object):
    def __init__(self, ordered_signals, batch_size, train_files, eval_files, test_files,
                 patient_specific='00',
                 min_queue_examples=128):
        self.signals = ordered_signals
        self.num_files = params.seizure_samples[params.input_length]
        self.ratio = params.seizure_ratio[params.input_length]
        self.train_files = train_files
        self.eval_files = eval_files
        self.test_files = test_files
        self.patient_specific = patient_specific
        if self.patient_specific == '00':
            self.dataset_dir = params.tfRecords_dir
            self.train_nums = self.num_files * len(train_files) * (1 + 4 * self.ratio)
            self.eval_nums = self.num_files * len(eval_files) * (1 + 4 * self.ratio)
            self.test_nums = self.num_files * len(test_files) * (1 + 4 * self.ratio)
        else:
            self.dataset_dir = params.tfRecords_dir + '/patient_specific/chb' + self.patient_specific
            self.train_nums = params.patient_specific[self.patient_specific][0]
            self.eval_nums = params.patient_specific[self.patient_specific][1]
            self.test_nums = params.patient_specific[self.patient_specific][2]
        self.data_prefix = "Standard_"
        self.labels = ["interictal", "preictal-I", "preictal-II", "preictal-III", "ictal"]
        self.batch_size = batch_size
        self.min_queue_examples = min_queue_examples
        self.dataset_build()

    def read_file(self, file_name):
        file_name_queue = tf.train.string_input_producer(file_name)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        reader_dict = {"label": tf.FixedLenFeature([], tf.string)}
        for signal in self.signals:
            reader_dict[self.data_prefix + signal] = tf.FixedLenFeature([], tf.string)
        features = tf.parse_single_example(serialized_example, features=reader_dict)
        feature_signals = []
        for signal in self.signals:
            feature = tf.decode_raw(features[self.data_prefix + signal], tf.float64)
            feature_signals.append(feature)
        signals = tf.concat(feature_signals, axis=0)
        signals = tf.cast(signals, tf.float32)
        signals = tf.reshape(signals, [18, params.epoches, 275])
        # 18x15x275
        siganls_PSD_frequency = signals[:, :, 0:128]
        # 128
        siganls_PSD_band = signals[:, :, 128:134]
        # 6
        siganls_Wavelet_frequency = signals[:, :, 134:261]
        # 127
        siganls_Wavelet_band = signals[:, :, 261:267]
        # 6
        signals_Time = signals[:, :, 268:275]
        # 7
        # mean, abs_mean, kurtosis, ptp, skewness, zero_crossing, std
        signals = tf.concat(
            [siganls_PSD_frequency, siganls_PSD_band, siganls_Wavelet_frequency, siganls_Wavelet_band, signals_Time],
            axis=2)
        signals = tf.reshape(signals, [18, params.epoches, 274])
        # 18x15x274
        simple_signals = tf.concat([siganls_Wavelet_band, signals_Time], axis=2)
        simple_signals = tf.reshape(simple_signals, [18, params.epoches, 13])
        # 18x15x19
        labels = tf.decode_raw(features["label"], tf.int64)
        labels = tf.reshape(labels, [1])
        # 1

        return self.augmentation(signals, simple_signals, labels)

    def augmentation(self, signals, simple_signals, labels):
        return signals, simple_signals, labels

    def tfrecords_to_batch(self, tfrecords_list, name):
        example_list = [self.read_file([i]) for i in tfrecords_list]
        signals_batch, simple_signals_batch, label_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=self.batch_size,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples,
            name=name
        )
        return {"signals": signals_batch, "simple_signals": simple_signals_batch, "labels": label_batch}

    def dataset_build(self):
        def add_dir_in_list(list, dir):
            if os.path.exists(dir):
                list.append(dir)
                return list
            else:
                raise Exception("The %s does not exists!" % dir)

        train_dir_list = []
        eval_dir_list = []
        test_dir_list = []
        for label in self.labels:
            for file in self.train_files:
                add_dir_in_list(train_dir_list, self.dataset_dir + "/%s/%s.tfrecords" % (label, file))
            for file in self.eval_files:
                add_dir_in_list(eval_dir_list, self.dataset_dir + "/%s/%s.tfrecords" % (label, file))
            for file in self.test_files:
                add_dir_in_list(test_dir_list, self.dataset_dir + "/%s/%s.tfrecords" % (label, file))
        self.train_batch = self.tfrecords_to_batch(train_dir_list, 'train')
        self.eval_batch = self.tfrecords_to_batch(eval_dir_list, 'eval')
        self.test_batch = self.tfrecords_to_batch(test_dir_list, 'test')


def test_input():
    import seaborn as sns
    D = ReadTFRecords(params.longitudinal_adjacent_signal, batch_size=32,
                      train_files=['00', '01', '02', '03', '04', '05', '06'],
                      eval_files=['07', '08'], test_files=['09'])
    print("Complete the building of DataSet")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        all_duration = 0
        for step in range(500):
            start = time.clock()
            train_signals, train_labels = sess.run([D.train_batch["signals"], D.train_batch["labels"]])
            duration = time.clock() - start
            all_duration += duration
            # eval_signals, eval_labels = sess.run([D.eval_batch["signals"], D.eval_batch["labels"]])
            # test_signals, test_labels = sess.run([D.test_batch["signals"], D.test_batch["labels"]])
            # train_signals = np.squeeze(train_signals)
            # f = open('/home/zengdifei/Output/CHB_MIT/test/test_input.txt', 'w')
            train_counts_label = {}
            # eval_counts_label = {}
            # test_counts_label = {}
            for label in range(5):
                train_counts_label[label] = 0
                # test_counts_label[label] = 0
                # eval_counts_label[label] = 0
            for i in range(D.batch_size):
                # print("The Sample %d with label : %d" % (i, labels[i]), file=f)
                train_counts_label[int(train_labels[i])] += 1
                # eval_counts_label[int(eval_labels[i])] += 1
                # test_counts_label[int(test_labels[i])] += 1
                # signal = np.squeeze(signals[i])
                # for j in range(signal.shape[0]):
                #     plt.plot(np.arange(0, params.epoches, 1), signal[j, :, 0] + j * 5, 'b')
                # plt.title(D.labels[int(labels[i])])
                # plt.savefig('/home/zengdifei/Output/CHB_MIT/test/signal_%03d.png' % i)
                # plt.close()
                # print("Signal :", signal, file=f)
                # corr = np.corrcoef(np.mean(signal, axis=2))
                # sns.heatmap(corr)
                # plt.title(D.labels[int(labels[i])])
                # plt.savefig('/home/zengdifei/Output/CHB_MIT/test/corr_%03d.png' % i)
                # plt.close()
                # print("Label :", labels[i], file=f)
            print("step %d: " % step, train_counts_label, "with run time: %.3fs" % duration)
            # print(eval_counts_label)
            # print(test_counts_label)
            # f.close()
        avg_duration = all_duration / 500
        print("tf.train.shuffle_batch_join run 500 steps with average duration: %.3fs" % avg_duration)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test_input()
