import random
import sys
import os
# sys.path.append("/home/zengdifei/Documents/CHB_MIT/src")
import preprocessing.label_wrapper as label_wrapper
import preprocessing.edf_extraction as edf_extraction
from datetime import timedelta
import utils.sharing_params as params


class Split(object):
    def __init__(self, patient):
        self.ratio = params.seizure_ratio[params.input_length]
        self.split_length = 2
        self.patient=patient
        # self.input_length=2
        self.ictal = []
        self._ictal_dict = {}
        self._ictal_all_duration_dict = {}
        self.preictal = {}
        # for i in range(3):
        #     self.preictal["%d" % (i + 1)] = []
        self.interictal = []
        self.interictal_not_in_train = {}
        #The interictal before a specifc ictal for later eval should not be contained in the trainning dataset
        self.dataset_dir = params.dataset_dir
        self.out_file_name = 'major_revison_post_pre_ictal_1h_len_%d/pred_v2' % (self.split_length)
        self._build_intervals()
        self.split = {}
        self.split["preictal"] = {}
        self.split["interictal"] = {}
        self._build_split()



    def file2dir(self, file):
        return self.dataset_dir + file[0:5] + "/" + file

    def _output_specific(self, label):
        if not os.path.exists(self.dataset_dir + \
                 self.out_file_name + \
                 '/patient_specific/split/'):
            os.makedirs(self.dataset_dir + \
                 self.out_file_name + \
                 '/patient_specific/split/')
        f = open(self.dataset_dir + \
                 self.out_file_name + \
                 '/patient_specific/split/' + label + ".txt", 'w')
        for tuple in self.split[label]:
            print(tuple[0], tuple[1], tuple[2], file=f)
        f.close()

    def _output(self, label):
        if not os.path.exists(self.dataset_dir + \
                 self.out_file_name + \
                 '/split/'):
            os.makedirs(self.dataset_dir + \
                 self.out_file_name + \
                 '/split/')
        f = open(self.dataset_dir + \
                 self.out_file_name + \
                 '/split/' + label + ".txt", 'w', )
        for tuple in self.split[label]:
            print(tuple[0], tuple[1], tuple[2], file=f)
        f.close()

    def _output_pred(self, split_list, patient, label, seizure_index):
        path=os.path.join(self.dataset_dir, self.out_file_name,'split', patient, str(seizure_index) )
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path +'/'+ label + ".txt", 'w', )
        for tuple in split_list:
            print(tuple[0], tuple[1], tuple[2], file=f)
        f.close()
    def _output_pred_len(self, split_list, patient, label):
        path=os.path.join(self.dataset_dir, self.out_file_name,'split_length', patient )
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path +'/'+ label + ".txt", 'a', )
        for tuple in split_list:
            print(tuple, split_list[tuple], file=f)
        f.close()
    def _output_pred_len_all(self, list_inter, list_pre, patient, label):
        path=os.path.join(self.dataset_dir, self.out_file_name,'split_length')
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path +'/'+ label + ".txt", 'a', )
        print('%s'%patient, file=f)
        print(list_inter, file=f)
        print(list_pre, file=f)
        # list_temp=[]
        # for tuple in list_inter:
        #     list_temp.append(list_inter[tuple])
        # print(list_temp, file=f)
        # list_temp = []
        # for tuple in list_pre:
        #     list_temp.append(list_pre[tuple])
        # print(list_temp, file=f)
        f.close()
    # def make_labels(self):
    #     print("ictal in %d seconds in %d segments :" % (self._ictal_duration, len(self.ictal)), self.ictal)
    #     print("interictal in %d seconds %d segments :" % (self._interictal_duration, len(self.interictal)),
    #           self.interictal)
    #     print("preictal-I in %d seconds %d segments :" % (self._preictal_I_duration, len(self.preictal["1"])),
    #           self.preictal["1"])
    #     print("preictal-II in %d seconds %d segments :" % (self._preictal_II_duration, len(self.preictal["2"])),
    #           self.preictal["2"])
    #     print("preictal-III in %d seconds %d segments :" % (self._preictal_III_duration, len(self.preictal["3"])),
    #           self.preictal["3"])
    #     self._output("ictal")
    #     self._output("interictal")
    #     self._output("preictal-I")
    #     self._output("preictal-II")
    #     self._output("preictal-III")

    def make_labels_specific(self):
        print("ictal in %d seconds in %d segments :" % (self._ictal_duration, len(self.ictal)), self.ictal)
        print("interictal in %d seconds %d segments :" % (self._interictal_duration, len(self.interictal)),
              self.interictal)
        print("preictal-I in %d seconds %d segments :" % (self._preictal_I_duration, len(self.preictal["1"])),
              self.preictal["1"])
        print("preictal-II in %d seconds %d segments :" % (self._preictal_II_duration, len(self.preictal["2"])),
              self.preictal["2"])
        print("preictal-III in %d seconds %d segments :" % (self._preictal_III_duration, len(self.preictal["3"])),
              self.preictal["3"])
        self._output_specific("ictal")
        self._output_specific("interictal")
        self._output_specific("preictal-I")
        self._output_specific("preictal-II")
        self._output_specific("preictal-III")

    def _get_sum(self, list):
        if len(list)==0:
            return 0
        else:
            return sum([x[2] - x[1] for x in list])/3600

    def _get_sum_pred(self, list):
        return sum([x[2] - x[1] for x in list])

    def check_file_channel(self, file):
        file_dir = self.file2dir(file)
        channels = edf_extraction.EdfFile(file_dir).get_channel_names()
        for channel in params.normal_signal:
            if not channel in channels:
                return False
        return True

    def _build_intervals(self):
        patient=self.patient
        filename = self.dataset_dir + "chb" + patient + "/chb" + patient + "-summary.txt"

        patient_label_wrapper = label_wrapper.LabelWrapper(filename)

        # seizure_list  [(file_name, st, en)]
        # file_list     [file_name]
        seizure_list, file_list = patient_label_wrapper.get_seizure_list()
        valid_seizure_list = []
        valid_file_list = []
        # 筛除通道不合法的文件
        # 在list中加入标准时间
        # valid_seizure_list    [(unix_st, unix_en, file_name, st, en)]
        # valid_file_list       [(unix_st, unix_en, file_name, 0, en]
        for seizure in seizure_list:
            if self.check_file_channel(seizure[0]):
                edf_file = edf_extraction.EdfFile(self.file2dir(seizure[0]))
                file_st = edf_file.get_start_datetime()
                st = file_st + timedelta(seconds=seizure[1])
                en = file_st + timedelta(seconds=seizure[2])
                valid_seizure_list.append([st, en, seizure[0], seizure[1], seizure[2]])
                del edf_file
        for file in file_list:
            if self.check_file_channel(file):
                edf_file = edf_extraction.EdfFile(self.file2dir(file))
                file_st = edf_file.get_start_datetime()
                file_en = edf_file.get_end_datetime()
                st = 0
                en = edf_file.get_file_duration()
                valid_file_list.append([file_st, file_en, file, st, en])
                del edf_file

        def takeFirst(elem):
            return elem[0]

        valid_seizure_list.sort(key=takeFirst)
        union_seizure_list = []
        seizure_times = 0
        # union_seizure_list    [(union_unix_st, union_unix_en, file_name, st, en)]
        # concatenate seizures with interval less than 60 mins
        for valid_seizure in valid_seizure_list:
            if seizure_times == 0 or union_seizure_list[seizure_times - 1][1] + timedelta(minutes=60) < \
                    valid_seizure[0]:
                union_seizure_list.append(valid_seizure)
                seizure_times += 1
            else:
                union_seizure_list[seizure_times - 1][1] = valid_seizure[1]
        print("patient %s :" % patient, seizure_times)

        #initialize self.interictal_not_in_train
        for seizure_index, union_seizure in enumerate(union_seizure_list):
            self.interictal_not_in_train["seizure-%d" % (seizure_index)]=[]



        # 生成ictal
        # for union_seizure in union_seizure_list:
        #     self.ictal.append([union_seizure[2], union_seizure[3], union_seizure[4]])

        # 生成preictal
        for seizure_index, union_seizure in enumerate(union_seizure_list):
            # for preictal in range(3):
            #     self.preictal["seizure-%d-pre%d" % (seizure_index, (preictal + 1))]=[]
            self.preictal["seizure-%d" % (seizure_index)] = []
            if seizure_index>=1:
                last_seizure_end=union_seizure_list[seizure_index-1][1]
            for file_index, valid_file in enumerate(valid_file_list):
                # if valid_file[]
                file_st = valid_file[0]
                file_en = valid_file[1]

                seizure_st = union_seizure[0]
                st = seizure_st - timedelta(minutes=30)
                en = seizure_st - timedelta(minutes=0)
                # st = seizure_st - timedelta(minutes=60 - preictal * 20)
                # en = seizure_st - timedelta(minutes=40 - preictal * 20)
                #caltulate the overlap
                cross_st = max(file_st, st)
                cross_en = min(file_en, en)
                if seizure_index >= 1:
                    cross_st = max(cross_st, last_seizure_end)
                #if the overlap is larger than self.split_length, the seg is included
                if cross_en > cross_st:
                    time_st = (cross_st - file_st).seconds
                    time_en = (cross_en - file_st).seconds
                    if time_en - time_st >= self.split_length:
                        self.preictal["seizure-%d" % (seizure_index)].append(
                            [valid_file[2], time_st, time_en])
                        if valid_file[2] == union_seizure[2]:
                            print(valid_file[2], "seizure%d" % (seizure_index), time_st,
                                  time_en,
                                  union_seizure[3])
                        else:
                            print(valid_file[2], "seizure%d" % (seizure_index), time_st,
                                  time_en,
                                  union_seizure[3] + valid_file[4], "from next edf file!")

        # 生成interictal
        interictal_hour=1
        for valid_file in valid_file_list:
            interictal_intervals = [[valid_file[0], valid_file[1]]]
            for union_seizure in (union_seizure_list):
                seizure_st = union_seizure[0]
                seizure_en = union_seizure[1]
                st = seizure_st - timedelta(hours=interictal_hour)
                en = seizure_en + timedelta(hours=interictal_hour)
                new_interictal_intervals = []
                for interval in interictal_intervals:
                    interval_st = interval[0]
                    interval_en = interval[1]
                    cross_st = max(interval_st, st)
                    cross_en = min(interval_en, en)
                    if cross_en > cross_st:
                        new_interictal_intervals.append([interval_st, cross_st])
                        new_interictal_intervals.append([cross_en, interval_en])
                    else:
                        # include the segs not having overlap with seisure +-4 hours
                        new_interictal_intervals.append(interval)
                interictal_intervals = []
                # delete the seg among seisure +-4 hours(3)
                for interval in new_interictal_intervals:
                    if interval[0] != interval[1]:
                        interictal_intervals.append(interval)
            for interval in interictal_intervals:
                time_st = (interval[0] - valid_file[0]).seconds
                time_en = (interval[1] - valid_file[0]).seconds
                if time_en - time_st >= self.split_length:
                    for seizure_index, union_seizure in enumerate(union_seizure_list):
                        # if seizure_index==0 and (interval[1]<=union_seizure[0]-timedelta(hours=interictal_hour)):
                        #     self.interictal_not_in_train["seizure-%d" % (0)].append([valid_file[2], time_st, time_en])
                        #     break
                        if (union_seizure[0]- timedelta(hours=interictal_hour))>=interval[1]:#and seizure_index<(len(union_seizure_list)-1)
                            self.interictal_not_in_train["seizure-%d" % (seizure_index)].append([valid_file[2], time_st, time_en])
                            break
                    # self.interictal.append([valid_file[2], time_st, time_en])
                    # print(valid_file[2], 'interictal', time_st, time_en)


        self._preictal_duration=[]
        self._interictal_duration = []
        # self._interictal_duration=[]
        for seizure in self.preictal:
            self._preictal_duration.append(self._get_sum(self.preictal[seizure]))
        for seizure in self.interictal_not_in_train:
            self._interictal_duration.append(self._get_sum(self.interictal_not_in_train[seizure]))
        print("preictal:", self._preictal_duration)
        print("interictal :", self._interictal_duration)
        self._output_pred_len_all(self._interictal_duration, self._preictal_duration, patient=patient,
                                  label='interictal_length')

        sum=0
        for i in self._preictal_duration:
            sum = sum + i
        self._preictal_duration_total=sum

        for i in self._interictal_duration:
            sum = sum + i
        self._interictal_duration_total=sum

    def _build_split(self):
        patient=self.patient
        self.sample_size={}
        self.sample_size["preictal"]={}
        self.sample_size["interictal"]={}
        overlap_stride=int(self.split_length/2)
        # interval_inter = self.interictal  # [seizure]
        for index, seizure in enumerate(self.preictal):
            interval_pre=self.preictal[seizure]

            self.split["preictal"]['sezure-%d'%index]=[]

            self.sample_size["preictal"]['sezure-%d'%index]=0

            for interval in interval_pre:
                for st in range(interval[1], interval[2] - self.split_length + 1, overlap_stride):
                    self.split["preictal"]['sezure-%d' % index].append((interval[0], st, st + self.split_length))
            # random.shuffle(self.split["ictal"])
            self.sample_size["preictal"]['sezure-%d'%index] = len(self.split["preictal"]['sezure-%d' % index])/3600
            self._output_pred(split_list=self.split["preictal"]['sezure-%d' % index], patient=patient, label='preictal', seizure_index=index)
        # self._output_pred_len(split_list=self.sample_size["preictal"], patient=patient, label='preictal_length')

        for index, seizure in enumerate(self.interictal_not_in_train):
            interval_pre = self.interictal_not_in_train[seizure]
            self.split["interictal"]['sezure-%d' % index] = []
            self.sample_size["interictal"]['sezure-%d' % index] = 0
            if len(interval_pre)>0:
                for interval in interval_pre:
                    for st in range(interval[1], interval[2] - self.split_length + 1, overlap_stride):
                        self.split["interictal"]['sezure-%d' % index].append((interval[0], st, st + self.split_length))
        # random.shuffle(self.split["ictal"])
            self.sample_size["interictal"]['sezure-%d'%index]  = len(self.split["interictal"]['sezure-%d' % index])/3600
            self._output_pred(split_list=self.split["interictal"]['sezure-%d' % index], patient=patient,
                          label='interictal', seizure_index=index)


        # self._output_pred_len(split_list=self.sample_size["interictal"], patient=patient, label='interictal_length')
          # self.sample_size["interictal"] = 0
        # for interval in self.interictal :
        #     for st in range(interval[1], interval[2] - self.split_length + 1, self._interictal_duration//self._preictal_duration_total*overlap_stride):
        #         self.split["interictal"].append((interval[0], st, st + self.split_length))
        # # random.shuffle(self.split["ictal"])
        # self.sample_size["interictal"] = len(self.split["interictal"])
        # self._output_pred(split_list=self.split["interictal"], patient=patient,
        #                   label='interictal', seizure_index='interictal')



if __name__ == "__main__":
    for patient in params.patients:#["03"]:
        split = Split(patient)
    # split.make_labels()
    # split.make_labels_specific()
