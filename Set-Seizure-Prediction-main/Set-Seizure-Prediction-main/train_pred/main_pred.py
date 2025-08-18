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
from train.loss_compute_pred import loss_compute
from tqdm import tqdm
import math

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
    # os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):


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
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if os.path.exists('checkpoints/' + args.exp_name+'/models/model.t7'):
        model.load_state_dict(torch.load('checkpoints/' + args.exp_name+'/models/model.t7'))
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr)#, momentum=0.9, weight_decay=5e-4)#, momentum=0.9, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # train_loader = DataLoader(CHB_data.load_data_loocv_specific_pred(partition='train'), num_workers=8,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(CHB_data.load_data_loocv_specific_pred(partition='test'), num_workers=8,
    #                          batch_size=args.batch_size, shuffle=False, drop_last=False)
    # eval_loader = DataLoader(CHB_data.load_data_loocv_specific_pred(partition='eval'), num_workers=8,
    #                          batch_size=args.batch_size, shuffle=False, drop_last=False)
    dataset_train=CHB_data.CHB_MIT_LOOCV_specific_pred_inter_split(partition='train', patient='01', seizure_time=7, seizure_for_test=3, rario_inter_pre=5, inter_for_test=2)
    dataset_test=CHB_data.CHB_MIT_LOOCV_specific_pred_inter_split\
        (partition='test', patient='01', seizure_time=7, seizure_for_test=3, rario_inter_pre=5, inter_for_test=2, mean=dataset_train.mean, std=dataset_train.std)

    train_loader = DataLoader(dataset_train,
        num_workers=8,
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test,
        num_workers=8,
        batch_size=args.batch_size, shuffle=False, drop_last=False)


    # scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.1, patience=20)
    # lambda1 = lambda epoch: 0.1 if epoch<80
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=learn_rate(args.epochs))

        # CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    # criterion = Func.cross_entropy()

    best_test_acc = 0

    def train_after(best_test_acc, loader, type):
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(loader, desc=type):
            data, label = data.to(device), label.to(device).squeeze()
            # data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            # multi_cal = loss_compute.multi_calculation(label, logits, args.preictal_fuzzification)
            # loss=multi_cal.loss
            try:
                loss = loss_compute(logits, label)
            except:
                pass
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if type=='test':
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                if best_test_acc>0.85:
                    torch.save(model.state_dict(), 'checkpoints/%s/models/model_%.4f.t7' % (args.exp_name, best_test_acc))

        outstr = '%s %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best_test: %.6f' % (type, epoch,
                                                                                             test_loss * 1.0 / count,
                                                                                             test_acc,
                                                                                             avg_per_class_acc,
                                                                                             best_test_acc)
        io.cprint(outstr)

        return best_test_acc

    for epoch in range(args.epochs):

        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        for data, label in tqdm(train_loader, desc="train"):
            data, label = data.to(device), label.to(device).squeeze()
            # data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)

            loss=loss_compute(logits, label)
            if math.isnan(loss):
                logits = model(data)
            preds = logits.max(dim=1)[1]
            loss.backward()
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        scheduler.step(metrics=train_loss)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f \n' \
                 'lr: %.10f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred), (opt.param_groups[0]['lr']))
        io.cprint(outstr)



        best_test_acc_new=train_after(best_test_acc, test_loader, 'test')
        # if (best_test_acc_new>best_test_acc and args.loocv!='True'):
        #     _=train_after(best_test_acc, eval_loader, 'eval')
        best_test_acc=best_test_acc_new
        ####################
        # Test
        ####################


# def test(args, io):
#     test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
#                              batch_size=args.test_batch_size, shuffle=True, drop_last=False)
#
#     device = torch.device("cuda" if args.cuda else "cpu")
#
#     # Try to load models
#     model = DGCNN(args).to(device)
#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(args.model_path))
#     model = model.eval()
#     test_acc = 0.0
#     count = 0.0
#     test_true = []
#     test_pred = []
#     for data, label in test_loader:
#         data, label = data.to(device), label.to(device).squeeze()
#         data = data.permute(0, 2, 1)
#         batch_size = data.size()[0]
#         logits = model(data)
#         preds = logits.max(dim=1)[1]
#         test_true.append(label.cpu().numpy())
#         test_pred.append(preds.detach().cpu().numpy())
#     test_true = np.concatenate(test_true)
#     test_pred = np.concatenate(test_pred)
#     test_acc = metrics.accuracy_score(test_true, test_pred)
#     avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
#     outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
#     io.cprint(outstr)


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
    parser.add_argument('--epochs', default=1000, help='the train epoch')
    # parser.add_argument('--lr_boundaries', default='80,200,250', help='the boundaries of learning rate')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--dropout', default=0.5, help='the probility to keep dropout')
    parser.add_argument('--MOVING_AVERAGE_DECAY', default=0.999, help='moving average decay')
    parser.add_argument('--use_batch_norm', default='1', help='whether or not use BN')
    parser.add_argument('--restore_step', default='0', help='the step used to restore')
    # parser.add_argument('--trainable', default='1', help='train or not')
    # parser.add_argument('--net_name', default='HGcnNet', help='the kind of net')
    parser.add_argument('--tree_classification', default='1', help='whether or not use Tree Classification')
    parser.add_argument('--preictal_fuzzification', default='1', help='whether or not use Preictal Fuzzification')
    parser.add_argument('--patient_specific', default='00', help='the number of patient specific')
    parser.add_argument('--train_set', default='', help='train set')
    parser.add_argument('--eval_set', default='', help='eval set')
    parser.add_argument('--test_set', default='', help='test set')
    parser.add_argument('--independent', type=str, default='False', help='individual independent or not')
    parser.add_argument('--overlap', type=str, default='True', help='overlap or not')
    parser.add_argument('--loocv', type=str, default='False', help='loocv or cross validation')
    args = parser.parse_args()

    args.num_classes = 2
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
        train(args, io)
    else:
        test(args, io)