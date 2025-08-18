from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import data.dataloader_pred as CHB_data
# from net.Net_torch import GcnNet, HGcnNet
import net.Net_torch_pred as network_multi
import numpy as np
from torch.utils.data import DataLoader
from utils.others import IOStream
import sklearn.metrics as metrics
from train_pred.loss_compute_pred import loss_compute
from tqdm import tqdm
import math
global datapath
datapath="/data/zhengruifeng/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/debug_77_preictal_1h_post_pre_ictal_1h_len_2/pred_v2/split/"
#"/data/zhengruifeng/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/77_preictal_1h_post_pre_ictal_4h_len_2/pred_v2/split/"
#'/dataset/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/pure_merge_preictal_30m_post_pre_ictal_1h_len_2/pred_v2/split/'
    #
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_pred.py checkpoints' + '/' + args.exp_name + '/' + 'train_pytorch.py.backup')
    os.system('cp ../net/Net_torch_pred.py checkpoints' + '/' + args.exp_name + '/' + 'Net_torch.py.backup')
    os.system('cp ../utils/gcn_utils_pred.py checkpoints' + '/' + args.exp_name + '/' + 'gcn_utils_torch.py.backup')
    os.system('cp ../utils/set_transformer_utils.py checkpoints' + '/' + args.exp_name + '/' + 'set_transformer_utils.py.backup')

def test(args, io):
    device = torch.device("cuda" if args.cuda else "cpu")
    # Try to load models
    if args.model == 'GcnNet':
        model = network_multi.GcnNet(args).to(device)

    elif args.model == 'HGcnNet':
        model = network_multi.HGcnNet(args).to(device)
    elif args.model == 'LT_hy_GcnNet':
        model = network_multi.LT_hybrid_GcnNet(args).to(device)
    elif args.model == 'SetTransformer':
        model = network_multi.SetTransformer(args).to(device)
    elif args.model == 'Gcn_Transformer':
        model = network_multi.Gcn_Transformer(args).to(device)
    # elif args.model == 'dgcnn_sta':
    #     model = DGCNN_STA(args).to(device)
    elif args.model == 'symmetric_GcnNet':
        model = network_multi.symmetric_GcnNet(args).to(device)
    elif args.model == 'tcn':
        channel_sizes = [10] * 3
        kernel_size = 4
        model = network_multi.Attention_TCN(44, 2, channel_sizes, kernel_size=kernel_size, dropout=0.6).to(device)
    elif args.model == 'All_Transformer':
        channel_sizes = [8, 4, 4]
        kernel_size = 4
        model = network_multi.All_Transformer(44, 2, channel_sizes, kernel_size=kernel_size, dropout=0.6, channel=args.channel).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load('checkpoints/%s/models/model_.t7' % (args.exp_name)))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # datapath = "/dataset/zhengruifeng/chb-mit-scalp-eeg-database-1.0.0/preictal_1h_post_ictal_30m_len_2/pred_v2/split/"
    seizure_list = os.listdir(os.path.join(datapath, args.patient))
    seizure_time = seizure_list.__len__()
    dataset_test = CHB_data.CHB_MIT_LOOCV_specific_pred \
        (datapath=datapath, partition='test', patient=args.patient, seizure_time=seizure_time,
         seizure_for_val=args.seizure_val, seizure_for_test=args.seizure_test)

        # CHB_data.CHB_MIT_LOOCV_specific_pred \
        # (datapath=datapath, partition='test', patient=args.patient, seizure_time=seizure_time,
        #  seizure_for_val=args.seizure_val, seizure_for_test=args.seizure_test)

    test_loader = DataLoader(dataset_test,
                             num_workers=8,
                             batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        # data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        # multi_cal = loss_compute.multi_calculation(label, logits, args.preictal_fuzzification)
        # loss=multi_cal.loss
        loss = loss_compute(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size

        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    # if os.path.exists('/home/zhengruifeng/PycharmProjects/EEG/CHB_MIT/src/train/checkpoints/tcn_test_02/result.txt'):
    #     os
    # with open('checkpoints/%s/result.txt' % (args.exp_name), 'w') as f:
    #     for item in test_true:
    #         print(item,file=f)
    with open('checkpoints/%s/pred_%s.txt' % (args.exp_name,args.channel), 'w') as f:
    # with open('/home/zhengruifeng/PycharmProjects/EEG/CHB_MIT/src/train/checkpoints/%s/result.txt'%(args.exp_name),'w') as f:
    #
    #     print('--------------------',file=f)
        for item in test_pred:
            print(item, file=f)

        test_pred_when_positive=test_pred[test_true==1]
        test_pred_when_negative = test_pred[test_true == 0]
        def counter_positive(input,desc):
            max_conti_positive = 0
            positive_count = 0
            max_index = 0
            long_conti_list=[]
            for index,item in enumerate(input):
                if item==1:
                    positive_count=positive_count+1
                    if positive_count> max_conti_positive:
                        max_conti_positive=positive_count
                        max_index=index

                else:
                    if positive_count > 200:
                        long_conti_list.append(str(positive_count))
                        long_conti_list.append(str(index))
                        long_conti_list.append(',')
                    positive_count=0
            print(str(desc)+': max_conti_positive %d, max_index %d'%(max_conti_positive,max_index), file=f)
            print(long_conti_list, file=f)
        counter_positive(test_pred_when_positive, 'pred_when_positive')
        counter_positive(test_pred_when_negative, 'pred_when_negative')
        for index, item in enumerate(test_true):
            if item==1:
                print('start_time: %d'%index, file=f)
                break
        # print(item,file=f)


    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)


    outstr = 'Test results: loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
                                                                                         test_loss * 1.0 / count,
                                                                                         test_acc,
                                                                                         avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='EEG')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='HGcnNet', metavar='N',
                        help='Model to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch_size', default=16, help='the size of examples in per batch')
    parser.add_argument('--epochs', default=50, help='the train epoch')
    # parser.add_argument('--lr_boundaries', default='80,200,250', help='the boundaries of learning rate')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--dropout', default=0.5, help='the probility to keep dropout')
    parser.add_argument('--MOVING_AVERAGE_DECAY', default=0.999, help='moving average decay')
    parser.add_argument('--use_batch_norm', default='1', help='whether or not use BN')
    parser.add_argument('--restore_step', default='0', help='the step used to restore')
    # parser.add_argument('--trainable', default='1', help='train or not')
    # parser.add_argument('--net_name', default='HGcnNet', help='the kind of net')
    # parser.add_argument('--tree_classification', default='1', help='whether or not use Tree Classification')
    # parser.add_argument('--preictal_fuzzification', default='1', help='whether or not use Preictal Fuzzification')
    # parser.add_argument('--patient_specific', default='00', help='the number of patient specific')
    parser.add_argument('--train_set', default='', help='train set')
    parser.add_argument('--eval_set', default='', help='eval set')
    parser.add_argument('--test_set', default='', help='test set')
    parser.add_argument('--independent', type=str, default='False', help='individual independent or not')
    parser.add_argument('--overlap', type=str, default='True', help='overlap or not')
    parser.add_argument('--loocv', type=str, default='False', help='loocv or cross validation')
    parser.add_argument('--patient', type=str, default='False', help='patient for analyze')
    parser.add_argument('--seizure_test', type=int, default=None, help='seizure for analyze')
    parser.add_argument('--seizure_val', type=int, default=None, help='seizure for analyze')
    parser.add_argument('--channel', type=str, default='empty', help='selected channel')
    args = parser.parse_args()

    args.num_classes = 2
    args.exp_name=os.path.join(args.exp_name,'patient'+args.patient, 'seizure'+str(args.seizure_test))
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        pass
    else:
        test(args, io)