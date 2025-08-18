import os
import random


def check(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def gen_dir(dir, index):
    output_dir = dir + index
    check(output_dir)
    summary_dir = output_dir + '/event'
    check(summary_dir)
    check(summary_dir + '/train')
    check(summary_dir + '/eval')
    restore_dir = output_dir + '/check'
    check(restore_dir)
    checkpoint_dir = output_dir + '/check/model.ckpt'
    visual_dir = output_dir + '/visual'
    check(visual_dir)
    return summary_dir, restore_dir, checkpoint_dir, visual_dir


def gen_split_files(visual_dir, train=None, eval=None, test=None):
    if os.path.exists(visual_dir + '/Test.txt'):
        f = open(visual_dir + '/Test.txt', 'r')
        line = f.readline().strip().split(':')[1].split(' [')
        f.close()

        def clean_files(line):
            return line[1:-2].split("', '")

        train_files = clean_files(line[1])
        eval_files = clean_files(line[2])
        test_files = clean_files(line[3])
        print('benchmark is :', train_files, eval_files, test_files)
    else:
        if train is None and eval is None and test is None:
            all_files = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
            train_files = random.sample(all_files, 7)
            for file in train_files: all_files.remove(file)
            eval_files = random.sample(all_files, 2)
            for file in eval_files: all_files.remove(file)
            test_files = random.sample(all_files, 1)
        else:
            train_files = train
            eval_files = eval
            test_files = test
        f = open(visual_dir + '/Test.txt', 'a')
        print('benchmark is :', train_files, eval_files, test_files, file=f)
        print('benchmark is :', train_files, eval_files, test_files)
        f.close()
    return train_files, eval_files, test_files

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()