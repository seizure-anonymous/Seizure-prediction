from sklearn.metrics import roc_curve, auc
import io
import os
import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt


def read_curve(dir, classes):
    f = io.open(dir, 'r')
    labels = []
    probabilities = []
    for line in f:
        line = line.strip().split()
        label = [float(a) for a in line[0:classes]]
        probability = [float(a) for a in line[classes:2 * classes]]
        labels.append(label)
        probabilities.append(probability)
    f.close()
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    return labels, probabilities


def compute_roc(labels, probabilities, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def make_roc_curves(curves, cla, save_dir):
    plt.figure()
    plt.rc('font', size=12)
    linewidth = 1
    if len(curves) == 6:
        colors = ['blue', 'cyan', 'gray', 'green', 'orange', 'red']
    elif len(curves) == 4:
        colors = ['blue', 'cyan', 'gray', 'red']
    elif len(curves) == 3:
        colors = ['blue', 'cyan', 'red']
    else:
        raise Exception('Wrong curves!')
    for i, curve in enumerate(curves):
        fpr = curve[0]
        tpr = curve[1]
        roc_auc = curve[2]
        name = curve[3]
        plt.plot(fpr[cla], tpr[cla], color=colors[i],
                 lw=linewidth, label='ROC curve of ' + name + ' (area = %0.4f)' % roc_auc[cla])
    # plt.plot([0, 1], [0, 1], color='black', lw=linewidth, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_dir)
    plt.close()


def draw_CHB_MIT_small():
    file_dir = "/home/zengdifei/Output/revision/CHB-MIT/ROC"
    classes = 5
    curves_name = ['GCN', 'm-HGCN', 'HGCN', 'GCN+TC', 'GCN+TC+PF', 'HGCN+TC+PF']
    curves = []
    for curve_name in curves_name:
        labels, probabilities = read_curve(os.path.join(file_dir, curve_name + '.txt'), classes)
        fpr, tpr, roc_auc = compute_roc(labels, probabilities, classes)
        curves.append([fpr, tpr, roc_auc, curve_name])
    rocs_name = ['interictal', 'preictal-I', 'preictal-II', 'preictal-III', 'ictal']
    for i, roc_name in enumerate(rocs_name):
        save_dir = os.path.join(file_dir, 'draw_CHB_MIT_small', roc_name + '.png')
        make_roc_curves(curves, i, save_dir)
    print("finish the drawing for small CHB_MIT!")


def draw_CHB_MIT_big():
    file_dir = "/home/zengdifei/Output/revision/CHB-MIT/ROC"
    classes = 5
    curves_name = ['GCN', 'HGCN', 'HGCN+TC+PF']
    curves = []
    for curve_name in curves_name:
        labels, probabilities = read_curve(os.path.join(file_dir, curve_name + '.txt'), classes)
        fpr, tpr, roc_auc = compute_roc(labels, probabilities, classes)
        if curve_name == 'HGCN+TC+PF':
            curve_name = 'HGCN+TC'
        curves.append([fpr, tpr, roc_auc, curve_name])
    rocs_name = ['interictal', 'preictal-I', 'preictal-II', 'preictal-III', 'ictal']
    for i, roc_name in enumerate(rocs_name):
        save_dir = os.path.join(file_dir, 'draw_CHB_MIT_big', roc_name + '.png')
        make_roc_curves(curves, i, save_dir)
    print("finish the drawing for big CHB_MIT!")


def draw_TUH_small():
    file_dir = "/home/zengdifei/Output/revision/TUH/ROC"
    classes = 3
    curves_name = ['GCN', 'm-HGCN', 'HGCN', 'HGCN+TC']
    curves = []
    for curve_name in curves_name:
        labels, probabilities = read_curve(os.path.join(file_dir, curve_name + '.txt'), classes)
        fpr, tpr, roc_auc = compute_roc(labels, probabilities, classes)
        curves.append([fpr, tpr, roc_auc, curve_name])
    rocs_name = ['interictal', 'preictal', 'ictal']
    for i, roc_name in enumerate(rocs_name):
        save_dir = os.path.join(file_dir, 'draw_TUH_small', roc_name + '.png')
        make_roc_curves(curves, i, save_dir)
    print("finish the drawing for small TUH!")


def draw_TUH_big():
    file_dir = "/home/zengdifei/Output/revision/TUH/ROC"
    classes = 3
    curves_name = ['GCN', 'HGCN', 'HGCN+TC']
    curves = []
    for curve_name in curves_name:
        labels, probabilities = read_curve(os.path.join(file_dir, curve_name + '.txt'), classes)
        fpr, tpr, roc_auc = compute_roc(labels, probabilities, classes)
        curves.append([fpr, tpr, roc_auc, curve_name])
    rocs_name = ['interictal', 'preictal', 'ictal']
    for i, roc_name in enumerate(rocs_name):
        save_dir = os.path.join(file_dir, 'draw_TUH_big', roc_name + '.png')
        make_roc_curves(curves, i, save_dir)
    print("finish the drawing for big TUH!")


if __name__ == "__main__":
    draw_CHB_MIT_small()
    draw_CHB_MIT_big()
    draw_TUH_small()
    draw_TUH_big()
