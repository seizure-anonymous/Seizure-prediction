import tensorflow as tf
import argparse
import os
import sys

sys.path.append("/home/zhengruifeng/PycharmProjects/EEG/CHB_MIT/src")
import preprocessing.input_process_dataset as input_process
import net.Net as Net
import train.loss_compute as loss_compute
import numpy as np
import time
import utils.others as utils_others
import utils.sharing_params as params
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

slim = tf.contrib.slim
parser = argparse.ArgumentParser(description='')
parser.add_argument('--GPU', default='0', help='the index of gpu')
parser.add_argument('--batch_size', default=16, help='the size of examples in per batch')
parser.add_argument('--epoch', default=1, help='the train epoch')
parser.add_argument('--lr_boundaries', default='80,200,250', help='the boundaries of learning rate')
parser.add_argument('--lr_values', default='0.1,0.01,0.001,0.0001', help='the values of learning_rate')
parser.add_argument('--dropout_keep_prob', default=0.5, help='the probility to keep dropout')
parser.add_argument('--MOVING_AVERAGE_DECAY', default=0.999, help='moving average decay')
parser.add_argument('--use_batch_norm', default='1', help='whether or not use BN')
parser.add_argument('--restore_step', default='0', help='the step used to restore')
parser.add_argument('--trainable', default='1', help='train or not')
parser.add_argument('--net_name', default='HGcnNet', help='the kind of net')
parser.add_argument('--tree_classification', default='1', help='whether or not use Tree Classification')
parser.add_argument('--preictal_fuzzification', default='1', help='whether or not use Preictal Fuzzification')
parser.add_argument('--patient_specific', default='00', help='the number of patient specific')
parser.add_argument('--train_set', default='', help='train set')
parser.add_argument('--eval_set', default='', help='eval set')
parser.add_argument('--test_set', default='', help='test set')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
args.batch_size = int(args.batch_size)
args.epoch = int(args.epoch)
lr_boundaries = []
for key in args.lr_boundaries.split(','):
    lr_boundaries.append(float(key))
lr_values = []
for key in args.lr_values.split(','):
    lr_values.append(float(key))
args.dropout_keep_prob = float(args.dropout_keep_prob)
args.use_batch_norm = bool(args.use_batch_norm == '1')
if bool(args.tree_classification == '1'):
    args.num_classes = 7
else:
    args.num_classes = 5
args.preictal_fuzzification = bool(args.preictal_fuzzification == '1')
args.train_set = args.train_set.split(',')
args.eval_set = args.eval_set.split(',')
args.test_set = args.test_set.rstrip().split(',')

summary_dir, restore_dir, checkpoint_dir, visual_dir = utils_others.gen_dir('/home/zhengruifeng/PycharmProjects/EEG/CHB_MIT/src/train/Output/CHB_MIT/',
                                                                            args.GPU)


def train():
    print(tf.__version__)

    with tf.Graph().as_default():
        if len(args.train_set) != 7:
            train_files, eval_files, test_files = utils_others.gen_split_files(visual_dir)
        else:
            train_files, eval_files, test_files = utils_others.gen_split_files(
                visual_dir,
                train=args.train_set,
                eval=args.eval_set,
                test=args.test_set,
            )



        Data = input_process.ReadTFRecords(params.normal_signal, args.batch_size, train_files, eval_files,
                                           test_files, args.patient_specific)
        # Data = input_process.ReadTFRecords(params.longitudinal_adjacent_signal, args.batch_size, train_files, eval_files,
        #                                    test_files, args.patient_specific)
        # Data = input_process.ReadTFRecords(params.transverse_adjacent_signal, args.batch_size, train_files, eval_files,
        #                                    test_files, args.patient_specific)

        signals = tf.placeholder(tf.float32, [args.batch_size, 18, params.epoches, 274])
        # signals = tf.placeholder(tf.float32, [args.batch_size, 18, params.epoch_length, 2])
        labels = tf.placeholder(tf.int64, [args.batch_size, 1])
        one_hot_labels = tf.one_hot(tf.reshape(labels, [args.batch_size]), 5)
        dropout_keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)

        channel_frequency_net = Net.Net(net_input=signals, net_name=args.net_name, num_classes=args.num_classes,
                                        is_training=is_training, dropout_keep_prob=dropout_keep_prob,
                                        use_batch_norm=args.use_batch_norm)
        logits = channel_frequency_net.end_points[channel_frequency_net.scope + '/logits']
        # time_dropout = channel_frequency_net.end_points[channel_frequency_net.scope + '/time_dropout']
        g = loss_compute.compute(labels, logits, args.preictal_fuzzification)
        slim.losses.add_loss(g.loss)
        total_loss = slim.losses.get_total_loss()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                      dtype=tf.int64)
        num_epoch = tf.cast(global_step, tf.float32) * args.batch_size / (Data.train_nums)
        lr = tf.train.piecewise_constant(num_epoch, boundaries=lr_boundaries, values=lr_values)
        with tf.name_scope('loss'):
            tf.summary.scalar('loss', g.loss)
            # tf.summary.scalar('self_loss', g.self_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('blur_predictions', g.blur_predictions)
            tf.summary.scalar('seizure_loss', g.seizure_loss)
            tf.summary.scalar('predict_loss', g.predict_loss)
            tf.summary.scalar('preictal_loss', g.preictal_loss)

            tf.summary.scalar('accuracy_5', g.accuracy_5)
            tf.summary.scalar('accuracy_3', g.accuracy_3)
            tf.summary.scalar('accuracy_2_predict', g.accuracy_2_predict)
            tf.summary.scalar('accuracy_2_seizure', g.accuracy_2_seizure)

            tf.summary.scalar('accuracy_ictal', g.accuracy_ictal)
            tf.summary.scalar('accuracy_preictal', g.accuracy_preictal)
            tf.summary.scalar('accuracy_preictalI', g.accuracy_preictalI)
            tf.summary.scalar('accuracy_preictalII', g.accuracy_preictalII)
            tf.summary.scalar('accuracy_preictalIII', g.accuracy_preictalIII)
            tf.summary.scalar('accuracy_interictal', g.accuracy_interictal)

            tf.summary.scalar('learning_rate', lr)
            print('Variables:')
            for variable in tf.trainable_variables():
                print(variable.name)
                # tf.summary.histogram(variable.name, variable)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            # GD
            train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss,
                                                                                      var_list=tf.trainable_variables(),
                                                                                      global_step=global_step,
                                                                                      name='GD')

        variables_averages = tf.train.ExponentialMovingAverage(args.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variables_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        saver_list = tf.global_variables()
        init = tf.global_variables_initializer()
        saver_restore = tf.train.Saver(saver_list)
        saver_train = tf.train.Saver(saver_list, max_to_keep=100)
        merged = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer_train = tf.summary.FileWriter(logdir=summary_dir + '/train', graph=sess.graph)
            summary_writer_eval = tf.summary.FileWriter(logdir=summary_dir + '/eval')

            ckpt = tf.train.get_checkpoint_state(restore_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if args.restore_step == '0':
                    temp_dir = ckpt.model_checkpoint_path
                else:
                    temp_dir = ckpt.model_checkpoint_path.split('-')[0] + '-' + args.restore_step
                temp_step = int(temp_dir.split('-')[1])
                print('Restore the global parameters in the step %d!' % (temp_step))
                saver_restore.restore(sess, temp_dir)
            else:
                print('Initialize the global parameters')
                init.run()

            eval_epoch = 0
            checkpoint_epoch = 0
            test_epoch = 0

            def train_once(sess):
                start = time.time()
                signals_batch, labels_batch = sess.run([Data.train_batch['signals'], Data.train_batch['labels']])
                process_time = time.time() - start
                feed_dict_train = {signals: signals_batch, labels: labels_batch, is_training: True,
                                   dropout_keep_prob: args.dropout_keep_prob}
                start_time = time.time()
                lr_value, summary, loss_value, step, epoch, _ = sess.run(
                    [lr, merged, g.loss, global_step, num_epoch, train_op],
                    feed_dict=feed_dict_train)
                end_time = time.time()
                sec_per_batch = end_time - start_time
                examples_per_sec = float(args.batch_size) / sec_per_batch
                summary_writer_train.add_summary(summary, int(1000 * epoch))
                pattern = 'epoch %.4f step %d with loss %.4f(%.1f examples/sec, %.3f process_time, %.3f train_time) in lr = %.4f'
                print(pattern % (epoch, step, loss_value, examples_per_sec, process_time, sec_per_batch, lr_value))
                return epoch, step

            def eval_once(sess, epoch):
                signals_batch, labels_batch = sess.run([Data.eval_batch['signals'], Data.eval_batch['labels']])
                feed_dict_eval = {signals: signals_batch, labels: labels_batch, is_training: False,
                                  dropout_keep_prob: 1.0}
                merged_value = sess.run(merged, feed_dict=feed_dict_eval)
                summary_writer_eval.add_summary(merged_value, (1000 * epoch))


            def test_once(sess, epoch):
                acc_2_seizure = []
                spec_2_seizure = []
                sens_2_seizure = []
                prec_2_seizure = []
                F1_2_seizure = []
                acc_2_predict = []
                spec_2_predict = []
                sens_2_predict = []
                prec_2_predict = []
                F1_2_predict = []
                acc_3 = []
                acc_5 = []
                confusion_matrix_2_seizure = []
                confusion_matrix_2_predict = []
                confusion_matrix_3 = []
                confusion_matrix_5 = []
                test_one_hot_labels = []
                # [B_all, 5]
                test_probabilities = []
                # [B_all, 5]
                durations = []
                for i in range(Data.test_nums // Data.batch_size):
                    signals_batch, labels_batch = sess.run([Data.test_batch['signals'], Data.test_batch['labels']])
                    feed_dict_eval = {signals: signals_batch, labels: labels_batch, is_training: False,
                                      dropout_keep_prob: 1.0}
                    acc_2_seizure_value, acc_2_predict_value, acc_3_value, acc_5_value, \
                    confusion_matrix_2_seizure_value, confusion_matrix_2_predict_value, \
                    confusion_matrix_3_value, confusion_matrix_5_value, \
                    spec_2_seizure_value, sens_2_seizure_value, prec_2_seizure_value, F1_2_seizure_value, \
                    spec_2_predict_value, sens_2_predict_value, prec_2_predict_value, F1_2_predict_value, \
                    one_hot_labels_batch, probabilities_batch = sess.run(
                        [g.accuracy_2_seizure, g.accuracy_2_predict, g.accuracy_3, g.accuracy_5,
                         g.confusion_matrix_2_seizure, g.confusion_matrix_2_predict,
                         g.confusion_matrix_3, g.confusion_matrix_5,
                         g.specificity_2_seizure, g.sensitivity_2_seizure, g.precision_2_seizure, g.F1_score_2_seizure,
                         g.specificity_2_predict, g.sensitivity_2_predict, g.precision_2_predict, g.F1_score_2_predict,
                         one_hot_labels, g.probabilities], feed_dict=feed_dict_eval
                    )
                    if np.isnan(acc_2_seizure_value) or np.isnan(spec_2_seizure_value) or \
                            np.isnan(sens_2_seizure_value) or np.isnan(prec_2_seizure_value) or \
                            np.isnan(F1_2_seizure_value) or np.isnan(acc_2_predict_value) or \
                            np.isnan(spec_2_predict_value) or np.isnan(sens_2_predict_value) or \
                            np.isnan(prec_2_predict_value) or np.isnan(F1_2_predict_value) or \
                            np.isnan(acc_3_value) or np.isnan(acc_5_value):
                        continue
                    acc_2_seizure.append(acc_2_seizure_value)
                    spec_2_seizure.append(spec_2_seizure_value)
                    sens_2_seizure.append(sens_2_seizure_value)
                    prec_2_seizure.append(prec_2_seizure_value)
                    F1_2_seizure.append(F1_2_seizure_value)
                    acc_2_predict.append(acc_2_predict_value)
                    spec_2_predict.append(spec_2_predict_value)
                    sens_2_predict.append(sens_2_predict_value)
                    prec_2_predict.append(prec_2_predict_value)
                    F1_2_predict.append(F1_2_predict_value)
                    acc_3.append(acc_3_value)
                    acc_5.append(acc_5_value)
                    confusion_matrix_2_seizure.append(confusion_matrix_2_seizure_value)
                    confusion_matrix_2_predict.append(confusion_matrix_2_predict_value)
                    confusion_matrix_3.append(confusion_matrix_3_value)
                    confusion_matrix_5.append(confusion_matrix_5_value)
                    test_one_hot_labels.append(one_hot_labels_batch)
                    test_probabilities.append(probabilities_batch)
                    start_time = time.time()
                    _ = sess.run(g.logits, feed_dict=feed_dict_eval)
                    duration = time.time() - start_time
                    durations.append(duration)

                durations = np.array(durations)
                average_duration = durations.mean()

                test_one_hot_labels = np.concatenate(test_one_hot_labels)
                test_probabilities = np.concatenate(test_probabilities)

                def normalize_confusion_matrix(mat):
                    """
                    normalization
                    :param mat: [n, n] np.float32
                    :return:
                    """
                    shape = mat.shape
                    if len(shape) != 2 or shape[0] != shape[1]:
                        raise Exception('The shape of mat is wrong!')
                    mat = mat.astype(np.float32)
                    for row in range(shape[0]):
                        if mat[row, :].sum() > 0:
                            mat[row, :] = mat[row, :] / mat[row, :].sum()
                    return mat

                def metric_confusion_matrix_2x2(mat):
                    """
                    metric
                    :param mat: [2, 2] np.float32
                    :return:
                    """
                    shape = mat.shape
                    if len(shape) != 2 or shape[0] != 2 or shape[1] != 2:
                        raise Exception('The shape of mat is wrong!')
                    mat = mat.astype(np.float32)
                    TP = mat[0, 0]
                    FN = mat[0, 1]
                    FP = mat[1, 0]
                    TN = mat[1, 1]
                    accuracy = (TP + TN) / (TP + FN + FP + TN)
                    specificity = TN / (TN + FP)
                    sensitivity = TP / (TP + FN)
                    precision = TP / (TP + FP)
                    F1_score = (2 * TP) / (2 * TP + FP + FN)
                    return (accuracy, specificity, sensitivity, precision, F1_score)

                epoch = int(epoch)
                f = open(visual_dir + '/Test.txt', 'a')
                # 2_seizure
                confusion_matrix_2_seizure = np.mean(
                    np.stack(confusion_matrix_2_seizure, axis=0).astype(np.float32), axis=0, keepdims=False
                )
                confusion_matrix_2_seizure = normalize_confusion_matrix(confusion_matrix_2_seizure)
                confusion_matrix_2_seizure_df = DataFrame(
                    confusion_matrix_2_seizure,
                    index=['non-seizure', 'seizure'],
                    columns=['non-seizure', 'seizure']
                )
                sns.heatmap(confusion_matrix_2_seizure_df, square=True, annot=True, fmt='.4f')
                title = 'confusion_matrix_2_seizure_%d' % epoch
                plt.title('confusion matrix of 2-class/seizure')
                plt.savefig(visual_dir + '/' + title + '.png')
                plt.close()
                print(
                    "Test 2_seizure in epoch %d with \naccuray : %.4f\nspecificity : %.4f\nsensitivity : %.4f\nprecision : %.4f\nF1_score : %.4f\n" %
                    ((epoch,) + metric_confusion_matrix_2x2(confusion_matrix_2_seizure)), file=f
                )
                # 2_predict
                confusion_matrix_2_predict = np.mean(
                    np.stack(confusion_matrix_2_predict, axis=0).astype(np.float32), axis=0, keepdims=False
                )
                confusion_matrix_2_predict = normalize_confusion_matrix(confusion_matrix_2_predict)
                confusion_matrix_2_predict_df = DataFrame(
                    confusion_matrix_2_predict,
                    index=['interictal', 'preictal'],
                    columns=['interictal', 'preictal']
                )
                sns.heatmap(confusion_matrix_2_predict_df, square=True, annot=True, fmt='.4f')
                title = 'confusion_matrix_2_predict_%d' % epoch
                plt.title('confusion matrix of 2-class/predict')
                plt.savefig(visual_dir + '/' + title + '.png')
                plt.close()
                print(
                    "Test 2_predict in epoch %d with \naccuray : %.4f\nspecificity : %.4f\nsensitivity : %.4f\nprecision : %.4f\nF1_score : %.4f\n" %
                    ((epoch,) + metric_confusion_matrix_2x2(confusion_matrix_2_predict)), file=f
                )
                # 3
                confusion_matrix_3 = np.mean(
                    np.stack(confusion_matrix_3, axis=0).astype(np.float32), axis=0, keepdims=False
                )
                confusion_matrix_3 = normalize_confusion_matrix(confusion_matrix_3)
                confusion_matrix_3_df = DataFrame(
                    confusion_matrix_3,
                    index=['interictal', 'preictal', 'ictal'],
                    columns=['interictal', 'preictal', 'ictal']
                )
                sns.heatmap(confusion_matrix_3_df, square=True, annot=True, fmt='.4f')
                title = 'confusion_matrix_3_%d' % epoch
                plt.title('confusion matrix of 3-class')
                plt.savefig(visual_dir + '/' + title + '.png')
                plt.close()
                # 5
                confusion_matrix_5 = np.mean(
                    np.stack(confusion_matrix_5, axis=0).astype(np.float32), axis=0, keepdims=False
                )
                confusion_matrix_5 = normalize_confusion_matrix(confusion_matrix_5)
                confusion_matrix_5_df = DataFrame(
                    confusion_matrix_5,
                    index=['interictal', 'prei-I', 'prei-II', 'prei-III', 'ictal'],
                    columns=['interictal', 'prei-I', 'prei-II', 'prei-III', 'ictal']
                )
                sns.heatmap(confusion_matrix_5_df, square=True, annot=True, fmt='.4f')
                title = 'confusion_matrix_5_%d' % epoch
                plt.title('confusion matrix of 5-class')
                plt.savefig(visual_dir + '/' + title + '.png')
                plt.close()
                # accuracy
                print('Test in epoch %d with acc_2_seizure %.4f' % (epoch, np.array(acc_2_seizure).mean()), file=f)
                print('Test in epoch %d with spec_2_seizure %.4f' % (epoch, np.array(spec_2_seizure).mean()), file=f)
                print('Test in epoch %d with sens_2_seizure %.4f' % (epoch, np.array(sens_2_seizure).mean()), file=f)
                print('Test in epoch %d with prec_2_seizure %.4f' % (epoch, np.array(prec_2_seizure).mean()), file=f)
                print('Test in epoch %d with F1_2_seizure %.4f' % (epoch, np.array(F1_2_seizure).mean()), file=f)
                print('Test in epoch %d with acc_2_predict %.4f' % (epoch, np.array(acc_2_predict).mean()), file=f)
                print('Test in epoch %d with spec_2_predict %.4f' % (epoch, np.array(spec_2_predict).mean()), file=f)
                print('Test in epoch %d with sens_2_predict %.4f' % (epoch, np.array(sens_2_predict).mean()), file=f)
                print('Test in epoch %d with prec_2_predict %.4f' % (epoch, np.array(prec_2_predict).mean()), file=f)
                print('Test in epoch %d with F1_2_predict %.4f' % (epoch, np.array(F1_2_predict).mean()), file=f)
                print('Test in epoch %d with acc_3 %.4f' % (epoch, np.array(acc_3).mean()), file=f)
                print('Test in epoch %d with acc_5 %.4f' % (epoch, np.array(acc_5).mean()), file=f)
                print('Test in epoch %d with average inference duration %.5fs' % (epoch, average_duration), file=f)
                print('------------------------------------------------------------------', file=f)
                # close file
                f.close()
                # record all probabilities of labels and predictions for AUC-ROC or AUC-PR
                f = open(visual_dir + '/Test_epoch%d.txt' % epoch, 'a')
                for i, _ in enumerate(test_one_hot_labels):
                    record_test = np.concatenate([test_one_hot_labels[i], test_probabilities[i]])
                    record_str = (record_test[0], record_test[1], record_test[2], record_test[3], record_test[4],
                                  record_test[5], record_test[6], record_test[7], record_test[8], record_test[9])
                    print("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % record_str, file=f)
                f.close()

            while (args.trainable == '1'):
                epoch, step = train_once(sess)
                #for debug
                #test_once(sess, epoch)
                #--------
                if epoch > eval_epoch + 0.1:
                    eval_once(sess, epoch)
                    eval_epoch = epoch
                if epoch > checkpoint_epoch + 100:
                    saver_train.save(sess, checkpoint_dir, global_step=step)
                    checkpoint_epoch = int(epoch)
                if epoch >= args.epoch - 150:
                    if epoch >= test_epoch + 10:
                        test_once(sess, epoch)
                        test_epoch = int(epoch)
                else:
                    if args.patient_specific == '00':
                        if epoch >= test_epoch + 1:
                            test_once(sess, epoch)
                            test_epoch = int(epoch)
                    else:
                        if epoch >= test_epoch + 25:
                            test_once(sess, epoch)
                            test_epoch = int(epoch)
                if epoch >= args.epoch:
                    break

            summary_writer_train.close()
            summary_writer_eval.close()
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
