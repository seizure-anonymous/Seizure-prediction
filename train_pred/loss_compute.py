import tensorflow as tf
import numpy as np
import utils.sharing_params as params


class compute(object):
    def __init__(self, labels, logits, preictal_fuzzification):
        self.labels = tf.squeeze(labels)
        self.preictal_fuzzification = preictal_fuzzification
        self.time_loss = []
        # T * [1]
        self.time_seizure_loss = []
        # T * [1]
        self.time_predict_loss = []
        # T * [1]
        self.time_preictal_loss = []
        # T * [1]
        self.time_predictions = []
        # T * [B]
        self.time_probabilities = []
        # T * [B, 5]
        self.dims = len(logits.shape.as_list())
        if self.dims == 2:
            self.logits = tf.expand_dims(logits, axis=1)
        elif self.dims == 3:
            self.logits = logits
        else:
            message = 'The dims %d of logits which should be 2 or 3 is wrong!' % self.dims
            raise Exception(message)

        # attention
        self.attention = tf.reduce_sum(tf.nn.tanh(self.logits), axis=2, keep_dims=False)
        (_, time) = self.attention.shape.as_list()
        self.attention = tf.nn.softmax(self.attention)
        # [B, T]

        (B, T, L) = self.logits.shape.as_list()
        for i in range(T):
            stacked_loss = self._loss_predictions(
                self.labels, tf.squeeze(self.logits[:, i, :]),
                # attention=tf.squeeze(self.attention[:, i])
            )
            self.time_loss.append(stacked_loss[0])
            self.time_predictions.append(stacked_loss[1])
            self.time_probabilities.append(stacked_loss[2])
            self.time_seizure_loss.append(stacked_loss[3])
            self.time_predict_loss.append(stacked_loss[4])
            self.time_preictal_loss.append(stacked_loss[5])

        self.loss = tf.stack(self.time_loss, axis=0)
        self.loss = tf.reduce_mean(self.loss, axis=0)

        self.seizure_loss = tf.stack(self.time_seizure_loss, axis=0)
        self.seizure_loss = tf.reduce_mean(self.seizure_loss, axis=0)

        self.predict_loss = tf.stack(self.time_predict_loss, axis=0)
        self.predict_loss = tf.reduce_mean(self.predict_loss, axis=0)

        self.preictal_loss = tf.stack(self.time_preictal_loss, axis=0)
        self.preictal_loss = tf.reduce_mean(self.preictal_loss, axis=0)

        # self.self_loss = []
        #
        # # Tx(T-1)/2 * [1]
        # for i in range(1, T):
        #     for j in range(i):
        #         i_j_loss_seizure = tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, i, 0:2]), axis=1), axis=1) - \
        #                            tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, j, 0:2]), axis=1), axis=1)
        #         i_j_loss_predict = tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, i, 2:4]), axis=1), axis=1) - \
        #                            tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, j, 2:4]), axis=1), axis=1)
        #         i_j_loss_preictal = tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, i, 4:7]), axis=1), axis=1) - \
        #                             tf.argmax(tf.nn.softmax(tf.squeeze(self.logits[:, j, 4:7]), axis=1), axis=1)
        #         # [B, num_classes]
        #         i_j_loss = tf.pow(tf.cast(i_j_loss_seizure, tf.float32), 2) + \
        #                    tf.pow(tf.cast(i_j_loss_predict, tf.float32), 2) + \
        #                    tf.pow(tf.cast(i_j_loss_preictal, tf.float32), 2)
        #         # [B]
        #         self.self_loss.append(tf.reduce_mean(i_j_loss, axis=0))
        # self.self_loss = tf.reduce_mean(tf.stack(self.self_loss, axis=0), axis=0)
        #
        # # add self_loss
        # # self.loss = self.loss + 0.5 * self.self_loss

        self.time_predictions = tf.stack(self.time_predictions, axis=1)
        # [B, T]
        # mode
        self.predictions, self.ratio_top = tf.py_func(mode, [self.time_predictions], [tf.int64, tf.float32])
        # self.predictions, self.ratio_top = tf.py_func(
        #     attention_mode, [self.time_predictions, self.attention], [tf.int64, tf.float32]
        # )
        self.predictions = tf.reshape(self.predictions, [B])
        self.ratio_top = tf.reshape(self.ratio_top, [B])
        self.blur_predictions = tf.py_func(where_count, [self.ratio_top], tf.float32)

        self.time_probabilities = tf.stack(self.time_probabilities, axis=1)
        # [B, T, 5]
        self.probabilities = tf.reduce_mean(self.time_probabilities, axis=1)
        # [B, 5]

        self.accuracy_5, self.accuracy_2_seizure, self.accuracy_2_predict, self.accuracy_3, \
        self.accuracy_ictal, self.accuracy_preictal, self.accuracy_preictalI, self.accuracy_preictalII, \
        self.accuracy_preictalIII, self.accuracy_interictal, self.confusion_matrix_2_seizure, \
        self.confusion_matrix_2_predict, self.confusion_matrix_3, self.confusion_matrix_5, \
        self.specificity_2_seizure, self.sensitivity_2_seizure, self.precision_2_seizure, self.F1_score_2_seizure, \
        self.specificity_2_predict, self.sensitivity_2_predict, self.precision_2_predict, self.F1_score_2_predict = self._accuracy(
            self.labels,
            self.predictions
        )

    def _loss_predictions(self, labels, logits,
                          seizure_ratio=params.seizure_loss_ratio,
                          predict_ratio=params.predict_loss_ratio,
                          preictal_ratio=params.preictal_loss_ratio,
                          attention=None):
        """
        :param labels: [B]
        :param logits: [B, num_classes]
        :return: loss, predictions
                 [1], [B], [1], [1], [1]
        """
        batch_size, num_classes = logits.shape.as_list()
        if attention is None:
            attention = tf.ones([batch_size], dtype=tf.float32, name='attention')
        else:
            if len(attention.shape) != 1 or attention.shape[0] != batch_size:
                raise Exception('The Shape of attention is wrong!')
        if num_classes == 5:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * attention
                , axis=0
            )
            seizure_loss = 0
            predict_loss = 0
            preictal_loss = 0
            probabilitys = tf.nn.softmax(logits)
            # [1]
            predictions = tf.argmax(probabilitys, axis=1)
            # [B, 1]
        elif num_classes == 7:
            seizure_probability = tf.nn.softmax(logits[:, 0:2], axis=1)
            predict_probability = tf.nn.softmax(logits[:, 2:4], axis=1)
            preictal_probability = tf.nn.softmax(logits[:, 4:7], axis=1)
            if self.preictal_fuzzification:
                seizure_labels, predict_labels, preictal_labels = tf.py_func(tree_label_with_PF, [labels],
                                                                             [tf.float32, tf.float32, tf.float32])
            else:
                seizure_labels, predict_labels, preictal_labels = tf.py_func(tree_label_without_PF, [labels],
                                                                             [tf.float32, tf.float32, tf.float32])
            seizure_loss = tf.reduce_mean(
                -tf.reduce_sum(seizure_labels * tf.log(seizure_probability),
                               axis=1) * attention,
                axis=0
            ) * seizure_ratio
            predict_loss = tf.reduce_mean(
                -tf.reduce_sum(predict_labels * tf.log(predict_probability),
                               axis=1) * attention,
                axis=0
            ) * predict_ratio
            preictal_loss = tf.reduce_mean(
                -tf.reduce_sum(preictal_labels * tf.log(preictal_probability),
                               axis=1) * attention,
                axis=0
            ) * preictal_ratio
            loss = seizure_loss + predict_loss + preictal_loss
            """
            The two methods to caculate predictions is nearly in accuracy.
            """
            """
            method 1
            """
            ictal_probability = seizure_probability[:, 1]
            preictalI_probability = seizure_probability[:, 0] * predict_probability[:, 1] * preictal_probability[:, 0]
            preictalII_probability = seizure_probability[:, 0] * predict_probability[:, 1] * preictal_probability[:, 1]
            preictalIII_probability = seizure_probability[:, 0] * predict_probability[:, 1] * preictal_probability[:, 2]
            interictal_probability = seizure_probability[:, 0] * predict_probability[:, 0]
            probabilitys = tf.stack(
                [interictal_probability, preictalI_probability, preictalII_probability, preictalIII_probability,
                 ictal_probability], axis=1
            )
            predictions = tf.argmax(probabilitys, axis=1)
            """
            method 2
            """
            # predictions = tf.py_func(tree_predictions, [seizure_probability, predict_probability, preictal_probability],
            #                          tf.int64)
            # predictions = tf.reshape(predictions, [batch_size])
        else:
            raise Exception('The num_classes of logits which should be 5 or 7 is wrong!')
        return loss, predictions, probabilitys, seizure_loss, predict_loss, preictal_loss

    def _accuracy(self, labels, predictions):
        """
        :param labels: [B]
        :param predictions: [B]
        :return: accuracys
                 10 * [1], [2, 2], [2, 2], [3, 3], [5, 5]
                 8 * [1]
        """
        return tf.py_func(error, [labels, predictions],
                          [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                           tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                           tf.int64, tf.int64, tf.int64, tf.int64,
                           tf.float32, tf.float32, tf.float32, tf.float32,
                           tf.float32, tf.float32, tf.float32, tf.float32]
                          )


def tree_predictions(seizure_probability, predict_probability, preictal_probability):
    """
    ictal           [1, ?, ?] -> 4
    preictal-III    [0, 1, 2] -> 3
    preictal-II     [0, 1, 1] -> 2
    preictal-I      [0, 1, 0] -> 1
    interictal      [0, 0, ?] -> 0
    :param seizure_probability: [B, 2] [non_seizure, seizure]
    :param predict_probability: [B, 2] [interictal, preictal]
    :param preictal_probability: [B, 3] [preictal-I, preicatl-II, preictal-III]
    :return: [B]
    """
    seizure_predictions = np.argmax(seizure_probability, axis=1)
    predict_predictions = np.argmax(predict_probability, axis=1)
    preictal_predictions = np.argmax(preictal_probability, axis=1)
    predictions = seizure_predictions * 4 + (1 - seizure_predictions) * predict_predictions * (preictal_predictions + 1)
    return predictions


def tree_label_with_PF(labels):
    """
    :param labels: [B]
    :return: seizure_labels, predict_labels, preictal_labels
             [B, 2], [B, 2], [B, 3]
    """
    label_dict = {
        0: [np.array([1, 0], dtype=np.float32),
            np.array([1, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32)],
        1: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0.9, 0.1, 0], dtype=np.float32)],
        2: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0.05, 0.9, 0.05], dtype=np.float32)],
        3: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0, 0.1, 0.9], dtype=np.float32)],
        4: [np.array([0, 1], dtype=np.float32),
            np.array([0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32)],
    }
    batch_size = labels.shape[0]
    seizure_labels = np.zeros([batch_size, 2], dtype=np.float32)
    predict_labels = np.zeros([batch_size, 2], dtype=np.float32)
    preictal_labels = np.zeros([batch_size, 3], dtype=np.float32)
    for i in range(batch_size):
        seizure_label, predict_label, preictal_label = label_dict[labels[i]]
        seizure_labels[i] = seizure_label
        predict_labels[i] = predict_label
        preictal_labels[i] = preictal_label
    return seizure_labels, predict_labels, preictal_labels


def tree_label_without_PF(labels):
    """
    :param labels: [B]
    :return: seizure_labels, predict_labels, preictal_labels
             [B, 2], [B, 2], [B, 3]
    """
    label_dict = {
        0: [np.array([1, 0], dtype=np.float32),
            np.array([1, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32)],
        1: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32)],
        2: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32)],
        3: [np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32)],
        4: [np.array([0, 1], dtype=np.float32),
            np.array([0, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32)],
    }
    batch_size = labels.shape[0]
    seizure_labels = np.zeros([batch_size, 2], dtype=np.float32)
    predict_labels = np.zeros([batch_size, 2], dtype=np.float32)
    preictal_labels = np.zeros([batch_size, 3], dtype=np.float32)
    for i in range(batch_size):
        seizure_label, predict_label, preictal_label = label_dict[labels[i]]
        seizure_labels[i] = seizure_label
        predict_labels[i] = predict_label
        preictal_labels[i] = preictal_label
    return seizure_labels, predict_labels, preictal_labels


def where_count(input):
    """
    count nums of right input
    :param input: [B]
    :return: nums
    """
    return np.where(input == 1, 1, 0).sum().astype(np.float32)


def attention_mode(input, attention):
    """
    :param input: [B, T]
    :param attention: [B, T]
    :return: [B], [B]
    """
    (batch_size, time) = input.shape
    modes = np.zeros([batch_size], dtype=np.int64)
    ratio_top = np.zeros([batch_size], dtype=np.float32)
    for i in range(batch_size):
        nums = input[i, :]
        att = attention[i, :]
        counts = np.zeros([5], dtype=np.float32)
        for t in range(time):
            counts[nums[t]] += att[t]
        modes[i] = np.argmax(counts)
        sort_counts = np.argsort(counts)[::-1]
        if sort_counts.shape[0] == 1:
            ratio_top[i] = 0
        else:
            ratio_top[i] = counts[sort_counts[1]] / counts[sort_counts[0]]
    return modes, ratio_top


def mode(input):
    """
    :param input: [B, T]
    :return: [B], [B]
    """
    (batch_size, time) = input.shape
    modes = np.zeros([batch_size], dtype=np.int64)
    ratio_top = np.zeros([batch_size], dtype=np.float32)
    for i in range(batch_size):
        counts = np.bincount(input[i, :])
        modes[i] = np.argmax(counts)
        sort_counts = np.argsort(counts)[::-1]
        if sort_counts.shape[0] == 1:
            ratio_top[i] = 0
        else:
            ratio_top[i] = counts[sort_counts[1]] / counts[sort_counts[0]]
    return modes, ratio_top


def tree_mode(input):
    """
    tree-predictions in time results
    this method is not alright
    0 -> 1,2,3
    4 -> 1,2,3
    :param input: [T, B]
    :return: [B]
    """
    batch_size = input.shape[1]
    modes = np.zeros([batch_size], dtype=np.int64)
    for i in range(batch_size):
        five_counts = np.bincount(input[:, i])
        five_counts_padding = np.zeros([5 - five_counts.shape[0]], dtype=np.int64)
        five_counts = np.concatenate((five_counts, five_counts_padding), axis=0)
        seizure_counts = np.array([five_counts[0:4].sum(), five_counts[4]], dtype=np.int64)
        predict_counts = np.array([five_counts[0], five_counts[1:4].sum()], dtype=np.int64)
        preictal_counts = five_counts[1:4]
        seizure_prediction = np.argmax(seizure_counts)
        predict_prediction = np.argmax(predict_counts)
        preictal_prediction = np.argmax(preictal_counts)
        modes[i] = seizure_prediction * 4 + (1 - seizure_prediction) * predict_prediction * (preictal_prediction + 1)
    return modes


def metric(TP, FN, FP, TN):
    """
    metric
    :param TP: True positive
    :param FN: False negative
    :param FP: False positive
    :param TN: True negative
    :return: accuracy, specificity, sensitivity, precision, F1_score
    """
    accuracy = (TP + TN) / (TP + FN + FP + TN+1e-6)
    specificity = TN / (TN + FP+1e-6)
    sensitivity = TP / (TP + FN+1e-6)
    precision = TP / (TP + FP+1e-6)
    F1_score = (2 * TP) / (2 * TP + FP + FN+1e-6)
    return accuracy.astype(np.float32), specificity.astype(np.float32), sensitivity.astype(np.float32), \
           precision.astype(np.float32), F1_score.astype(np.float32)


def error(labels, predicts):
    """
    :param labels: [B]
    :param predicts: [B]
    :return: accuracys
             10 * [1], [2, 2], [2, 2], [3, 3], [5, 5]
             8 * [1]
    """
    batch_size = labels.shape[0]
    err = np.zeros((5, 5), dtype=np.int64)
    for i in range(batch_size):
        label = labels[i]
        predict = predicts[i]
        err[label][predict] += 1
    err = err.astype(np.float32) / batch_size

    # 5类准确率 ictal/preictal-I~III/interictal
    accuracy_five = np.array([err[i][i] for i in range(5)], dtype=np.float32).sum()

    # 2类 seizure/non-seizure

    TP_two_for_seizure = np.array([err[0][0], err[0][1], err[0][2], err[0][3],
                                   err[1][0], err[1][1], err[1][2], err[1][3],
                                   err[2][0], err[2][1], err[2][2], err[2][3],
                                   err[3][0], err[3][1], err[3][2], err[3][3]], dtype=np.float32).sum()
    FN_two_for_seizure = np.array([err[0][4], err[1][4], err[2][4], err[3][4]], dtype=np.float32).sum()
    FP_two_for_seizure = np.array([err[4][0], err[4][1], err[4][2], err[4][3]], dtype=np.float32).sum()
    TN_two_for_seizure = np.array([err[4][4]], dtype=np.float32).sum()
    accuracy_two_for_seizure, specificity_two_for_seizure, sensitivity_two_for_seizure, precision_two_for_seizure, \
    F1_score_two_for_seizure = metric(TP_two_for_seizure, FN_two_for_seizure, FP_two_for_seizure, TN_two_for_seizure)

    # accuracy_two_for_seizure = np.array([err[0][0], err[0][1], err[0][2], err[0][3],
    #                                      err[1][0], err[1][1], err[1][2], err[1][3],
    #                                      err[2][0], err[2][1], err[2][2], err[2][3],
    #                                      err[3][0], err[3][1], err[3][2], err[3][3],
    #                                      err[4][4]], dtype=np.float32).sum()

    # 2类准确率 preictal/interictal
    TP_two_for_predict = np.array([err[0][0]], dtype=np.float32).sum()
    FN_two_for_predict = np.array([err[0][1], err[0][2], err[0][3]], dtype=np.float32).sum()
    FP_two_for_predict = np.array([err[1][0], err[2][0], err[3][0]], dtype=np.float32).sum()
    TN_two_for_predict = np.array([err[1][1], err[1][2], err[1][3],
                                   err[2][1], err[2][2], err[2][3],
                                   err[3][1], err[3][2], err[3][3]], dtype=np.float32).sum()
    accuracy_two_for_predict, specificity_two_for_predict, sensitivity_two_for_predict, precision_two_for_predict, \
    F1_score_two_for_predict = metric(TP_two_for_predict, FN_two_for_predict, FP_two_for_predict, TN_two_for_predict)
    # accuracy_two_for_predict = np.array([err[0][0],
    #                                      err[1][1], err[1][2], err[1][3],
    #                                      err[2][1], err[2][2], err[2][3],
    #                                      err[3][1], err[3][2], err[3][3]],
    #                                     dtype=np.float32).sum() / \
    #                            np.array([err[0][0], err[0][1], err[0][2], err[0][3], err[0][4],
    #                                      err[1][0], err[1][1], err[1][2], err[1][3], err[1][4],
    #                                      err[2][0], err[2][1], err[2][2], err[2][3], err[2][4],
    #                                      err[3][0], err[3][1], err[3][2], err[3][3], err[3][4]],
    #                                     dtype=np.float32).sum()

    # 3类准确率 ictal/preictal/interictal
    accuracy_three = np.array([err[0][0],
                               err[1][1], err[1][2], err[1][3],
                               err[2][1], err[2][2], err[2][3],
                               err[3][1], err[3][2], err[3][3],
                               err[4][4]], dtype=np.float32).sum()
    tiny=np.float32(1e-6)
    # ictal准确率
    accuracy_ictal = np.array([err[4][4]], dtype=np.float32).sum() / \
                     (np.array([err[4][0], err[4][1], err[4][2], err[4][3], err[4][4]], dtype=np.float32).sum()+tiny)

    # preictal准确率
    accuracy_preictal = np.array([err[1][1], err[1][2], err[1][3],
                                  err[2][1], err[2][2], err[2][3],
                                  err[3][1], err[3][2], err[3][3]],
                                 dtype=np.float32).sum() / \
                        (np.array([err[1][0], err[1][1], err[1][2], err[1][3], err[1][4],
                                  err[2][0], err[2][1], err[2][2], err[2][3], err[2][4],
                                  err[3][0], err[3][1], err[3][2], err[3][3], err[3][4]],
                                          dtype=np.float32).sum()+tiny)

    # preictal-I准确率
    accuracy_preictalI = np.array([err[3][3]], dtype=np.float32).sum() / \
                         (np.array([err[3][0], err[3][1], err[3][2], err[3][3], err[3][4]], dtype=np.float32).sum()+tiny)

    # preictal-II准确率
    accuracy_preictalII = np.array([err[2][2]], dtype=np.float32).sum() / \
                          (np.array([err[2][0], err[2][1], err[2][2], err[2][3], err[2][4]], dtype=np.float32).sum()+tiny)

    # preictal-III准确率
    accuracy_preictalIII = np.array([err[1][1]], dtype=np.float32).sum() / \
                           (np.array([err[1][0], err[1][1], err[1][2], err[1][3], err[1][4]], dtype=np.float32).sum()+tiny)

    # interictal准确率
    accuracy_interictal = np.array([err[0][0]], dtype=np.float32).sum() / \
                          (np.array([err[0][0], err[0][1], err[0][2], err[0][3], err[0][4]], dtype=np.float32).sum()+tiny)

    # confusion_matrix_5
    label_trans_to_5 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    confusion_matrix_5 = np.zeros((5, 5), dtype=np.int64)
    for i in range(batch_size):
        label = label_trans_to_5[labels[i]]
        predict = label_trans_to_5[predicts[i]]
        confusion_matrix_5[label][predict] += 1

    # confusion_matrix_3
    label_trans_to_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    confusion_matrix_3 = np.zeros((3, 3), dtype=np.int64)
    for i in range(batch_size):
        label = label_trans_to_3[labels[i]]
        predict = label_trans_to_3[predicts[i]]
        confusion_matrix_3[label][predict] += 1

    # confusion_matrix_2_predict
    label_trans_to_2_predict = {0: 0, 1: 1, 2: 1, 3: 1}
    confusion_matrix_2_predict = np.zeros((2, 2), dtype=np.int64)
    for i in range(batch_size):
        if labels[i] == 4 or predicts[i] == 4:
            continue
        label = label_trans_to_2_predict[labels[i]]
        predict = label_trans_to_2_predict[predicts[i]]
        confusion_matrix_2_predict[label][predict] += 1

    # confusion_matrix_2_seizure
    label_trans_to_2_seizure = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
    confusion_matrix_2_seizure = np.zeros((2, 2), dtype=np.int64)
    for i in range(batch_size):
        label = label_trans_to_2_seizure[labels[i]]
        predict = label_trans_to_2_seizure[predicts[i]]
        confusion_matrix_2_seizure[label][predict] += 1

    return accuracy_five, accuracy_two_for_seizure, accuracy_two_for_predict, accuracy_three, \
           accuracy_ictal, accuracy_preictal, accuracy_preictalI, accuracy_preictalII, accuracy_preictalIII, accuracy_interictal, \
           confusion_matrix_2_seizure, confusion_matrix_2_predict, confusion_matrix_3, confusion_matrix_5, \
           specificity_two_for_seizure, sensitivity_two_for_seizure, precision_two_for_seizure, F1_score_two_for_seizure, \
           specificity_two_for_predict, sensitivity_two_for_predict, precision_two_for_predict, F1_score_two_for_predict
