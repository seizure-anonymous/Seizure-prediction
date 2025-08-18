import numpy as np
import utils.sharing_params as params
import torch
import torch.nn.functional as Func
import torch.nn as nn

class loss_compute(nn.Module):
    def __init__(self, preictal_fuzzification=True):
        super(loss_compute, self).__init__()
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
        self.preictal_fuzzification=preictal_fuzzification
        # self.logits_shape=(0,0,0)
    def _loss_predictions(self, labels, logits, preictal_fuzzification,
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
        batch_size, num_classes = logits.shape
        if attention is None:
            attention = torch.ones(batch_size, dtype=torch.float32)
        else:
            if len(attention.shape) != 1 or attention.shape[0] != batch_size:
                raise Exception('The Shape of attention is wrong!')
        if num_classes == 5:
            loss = torch.mean(
                Func.cross_entropy(target=labels, input=logits) * attention
                , dim=0
            )  # tf.nn.sparse_softmax_cross_entropy_with_logits
            seizure_loss = 0
            predict_loss = 0
            preictal_loss = 0
            probabilitys = Func.softmax(logits)
            # [1]
            predictions = torch.argmax(probabilitys, dim=1)  # tf.argmax(probabilitys, axis=1)
            # [B, 1]
        elif num_classes == 7:
            seizure_probability = Func.softmax(logits[:, 0:2], dim=1)
            predict_probability = Func.softmax(logits[:, 2:4], dim=1)
            preictal_probability = Func.softmax(logits[:, 4:7], dim=1)
            if preictal_fuzzification:
                seizure_labels, predict_labels, preictal_labels = tree_label_with_PF(labels)
            else:
                seizure_labels, predict_labels, preictal_labels = tree_label_without_PF(labels)
            seizure_labels = seizure_labels.to(seizure_probability.device)
            predict_labels = predict_labels.to(seizure_probability.device)
            preictal_labels = preictal_labels.to(seizure_probability.device)
            attention = attention.to(seizure_probability.device)
            seizure_loss = torch.mean(
                -torch.sum(seizure_labels * torch.log(seizure_probability+1e-7),
                           dim=1) * attention,
                dim=0
            ) * seizure_ratio
            predict_loss = torch.mean(
                -torch.sum(predict_labels * torch.log(predict_probability+1e-7),
                           dim=1) * attention,
                dim=0
            ) * predict_ratio
            preictal_loss = torch.mean(
                -torch.sum(preictal_labels * torch.log(preictal_probability+1e-7),
                           dim=1) * attention,
                dim=0
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
            probabilitys = torch.stack(
                [interictal_probability, preictalI_probability, preictalII_probability, preictalIII_probability,
                 ictal_probability], dim=1
            )
            predictions = torch.argmax(probabilitys, dim=1)
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
        return error(labels, predictions)
    #called after forward running
    def multi_cal(self, logits, labels):
        with torch.no_grad():
            (B, T, L) = logits.shape
            seizure_loss = torch.stack(self.time_seizure_loss, dim=0)
            seizure_loss = torch.mean(seizure_loss, dim=0)

            predict_loss = torch.stack(self.time_predict_loss, dim=0)
            predict_loss = torch.mean(predict_loss, dim=0)

            preictal_loss = torch.stack(self.time_preictal_loss, dim=0)
            preictal_loss = torch.mean(preictal_loss, dim=0)

            time_predictions = torch.stack(self.time_predictions, dim=1)
            # [B, T]
            # mode
            predictions, ratio_top = mode(time_predictions)
            # self.predictions, self.ratio_top = tf.py_func(
            #     attention_mode, [self.time_predictions, self.attention], [tf.int64, tf.float32]
            # )
            predictions = predictions.reshape(B)  # tf.reshape(self.predictions, [B])
            ratio_top = ratio_top.reshape(B)  # tf.reshape(self.ratio_top, [B])
            blur_predictions = where_count(ratio_top)

            time_probabilities = torch.stack(self.time_probabilities, dim=1)
            # [B, T, 5]
            probabilities = torch.mean(time_probabilities, dim=1)
            # [B, 5]

            accuracy_5, accuracy_2_seizure, accuracy_2_predict, accuracy_3, \
            accuracy_ictal, accuracy_preictal, accuracy_preictalI, accuracy_preictalII, \
            accuracy_preictalIII, accuracy_interictal, confusion_matrix_2_seizure, \
            confusion_matrix_2_predict, confusion_matrix_3, confusion_matrix_5, \
            specificity_2_seizure, sensitivity_2_seizure, precision_2_seizure, F1_score_2_seizure, \
            specificity_2_predict, sensitivity_2_predict, precision_2_predict, F1_score_2_predict = self._accuracy(
                labels,
                predictions
            )
            return predictions, seizure_loss, predict_loss, preictal_loss, blur_predictions, probabilities,\
                accuracy_5, accuracy_2_seizure, accuracy_2_predict, accuracy_3, \
            accuracy_ictal, accuracy_preictal, accuracy_preictalI, accuracy_preictalII, \
            accuracy_preictalIII, accuracy_interictal, confusion_matrix_2_seizure, \
            confusion_matrix_2_predict, confusion_matrix_3, confusion_matrix_5, \
            specificity_2_seizure, sensitivity_2_seizure, precision_2_seizure, F1_score_2_seizure, \
            specificity_2_predict, sensitivity_2_predict, precision_2_predict, F1_score_2_predict

    def forward(self, logits, labels):
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
        # time_loss=[]
        dims = len(logits.shape)
        if dims == 2:
            logits = logits.unsqueeze(1)  # tf.expand_dims(logits, axis=1)
        elif dims == 3:
            logits = logits
        else:
            message = 'The dims %d of logits which should be 2 or 3 is wrong!' % self.dims
            raise Exception(message)

        # attention
        attention = torch.sum(torch.tanh(logits), dim=2,
                                   keepdim=False)  # tf.reduce_sum(tf.nn.tanh(self.logits), axis=2, keep_dims=False)
        (_, time) = attention.shape
        attention = Func.softmax(attention, dim=1)  # tf.nn.softmax(self.attention)
        # [B, T]

        (B, T, L) = logits.shape
        self.logits_shape=(B, T, L)
        for i in range(T):
            stacked_loss = self._loss_predictions(
                labels, (logits[:, i, :]), self.preictal_fuzzification
                # attention=tf.squeeze(self.attention[:, i])
            )
            self.time_loss.append(stacked_loss[0])
            self.time_predictions.append(stacked_loss[1])
            self.time_probabilities.append(stacked_loss[2])
            self.time_seizure_loss.append(stacked_loss[3])
            self.time_predict_loss.append(stacked_loss[4])
            self.time_preictal_loss.append(stacked_loss[5])

        loss = torch.stack(self.time_loss, dim=0)  # tf.stack(self.time_loss, axis=0)
        loss = torch.mean(loss, dim=0, keepdim=False)
        return loss

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
    batch_size = labels.shape[0]
    seizure_labels = torch.zeros([batch_size, 2], dtype=torch.float32)
    predict_labels = torch.zeros([batch_size, 2], dtype=torch.float32)
    preictal_labels = torch.zeros([batch_size, 3], dtype=torch.float32)
    for i in range(batch_size):
        if labels[i] ==0:
            seizure_label, predict_label, preictal_label=torch.Tensor([1,0]), torch.Tensor([1,0]),torch.Tensor([0,0,0])
        elif labels[i]==1:
            seizure_label, predict_label, preictal_label = torch.Tensor([1, 0]), torch.Tensor([0, 1]), torch.Tensor([0.9, 0.1, 0])
        elif labels[i]==2:
            seizure_label, predict_label, preictal_label = torch.Tensor([1, 0]), torch.Tensor([0, 1]), torch.Tensor([0.05, 0.9, 0.05])
        elif labels[i]==3:
            seizure_label, predict_label, preictal_label = torch.Tensor([1, 0]), torch.Tensor([0, 1]), torch.Tensor([0, 0.1, 0.9])
        elif labels[i]==4:
            seizure_label, predict_label, preictal_label = torch.Tensor([0, 1]), torch.Tensor([0, 0]), torch.Tensor([0, 0, 0])
        else:
            raise exit("wrong labels")
        # seizure_label, predict_label, preictal_label = label_dict[labels[i]]
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
    modes = torch.zeros([batch_size], dtype=torch.int64)
    ratio_top = torch.zeros([batch_size], dtype=torch.float32)
    for i in range(batch_size):
        counts = torch.bincount(input[i, :])
        modes[i] = torch.argmax(counts)
        sort_counts = torch.argsort(counts)#[::-1]
        if sort_counts.shape[0] == 1:
            ratio_top[i] = 0
        else:
            ratio_top[i] = counts[sort_counts[-2]] / counts[sort_counts[-1]]
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
        label = label_trans_to_5[int(labels[i])]
        predict = label_trans_to_5[int(predicts[i])]
        confusion_matrix_5[label][predict] += 1

    # confusion_matrix_3
    label_trans_to_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    confusion_matrix_3 = np.zeros((3, 3), dtype=np.int64)
    for i in range(batch_size):
        label = label_trans_to_3[int(labels[i])]
        predict = label_trans_to_3[int(predicts[i])]
        confusion_matrix_3[label][predict] += 1

    # confusion_matrix_2_predict
    label_trans_to_2_predict = {0: 0, 1: 1, 2: 1, 3: 1}
    confusion_matrix_2_predict = np.zeros((2, 2), dtype=np.int64)
    for i in range(batch_size):
        if labels[i] == 4 or predicts[i] == 4:
            continue
        label = label_trans_to_2_predict[int(labels[i])]
        predict = label_trans_to_2_predict[int(predicts[i])]
        confusion_matrix_2_predict[label][predict] += 1

    # confusion_matrix_2_seizure
    label_trans_to_2_seizure = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
    confusion_matrix_2_seizure = np.zeros((2, 2), dtype=np.int64)
    for i in range(batch_size):
        label = label_trans_to_2_seizure[int(labels[i])]
        predict = label_trans_to_2_seizure[int(predicts[i])]
        confusion_matrix_2_seizure[label][predict] += 1

    return accuracy_five, accuracy_two_for_seizure, accuracy_two_for_predict, accuracy_three, \
           accuracy_ictal, accuracy_preictal, accuracy_preictalI, accuracy_preictalII, accuracy_preictalIII, accuracy_interictal, \
           confusion_matrix_2_seizure, confusion_matrix_2_predict, confusion_matrix_3, confusion_matrix_5, \
           specificity_two_for_seizure, sensitivity_two_for_seizure, precision_two_for_seizure, F1_score_two_for_seizure, \
           specificity_two_for_predict, sensitivity_two_for_predict, precision_two_for_predict, F1_score_two_for_predict
