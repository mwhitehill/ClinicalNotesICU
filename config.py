import os
import sys
import socket

# Make sure can find the other repos
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'mimic3-benchmarks'))
sys.path.append(os.getcwd())

class Config():
    def __init__(self):
        self.basepath = './'
        self.save_path = r'Z:\classes\cse517\project\ClinicalNotesICU' if socket.gethostname() != 'area51.cs.washington.edu' else  r'/data/ClinicalNotesICU'
        self.data = '../mimic3-benchmarks/data/' if socket.gethostname() != 'area51.cs.washington.edu' else r'/data/ClinicalNotesICU'
        self.timestep = 1.0
        self.normalizer_state = os.path.join(self.basepath, 'ihm_ts1.0.input_str-previous.start_time-zero.normalizer')
        self.imputation = 'previous'
        self.small_part = False
        self.textdata = self.basepath + 'text/'
        self.embeddingspath = '../BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.buffer_size = 100
        self.learning_rate = 1e-4
        self.max_len = 1000
        self.break_text_at = 300
        self.padding_type = 'Zero'
        self.los_path = os.path.join(self.data, 'length-of-stay')
        self.decompensation_path = os.path.join(self.data, 'length-decompensation-stay')
        self.ihm_path = os.path.join(self.data, 'in-hospital-mortality')
        self.textdata_fixed = os.path.join(self.data, 'root', 'train_text_fixed')
        self.multitask_path = os.path.join(self.data, 'multitask')
        self.starttime_path = os.path.join(self.data, 'starttime.pkl')
        self.rnn_hidden_units = 256
        self.maximum_number_events = 150
        self.conv1d_channel_size = 256
        self.test_textdata_fixed = os.path.join(self.data, 'root', 'test_text_fixed')
        self.test_starttime_path = os.path.join(self.data, 'test_starttime.pkl')
        self.dropout = 0.9
        self.index2word_path = os.path.join(self.basepath, 'index2word.pkl') if socket.gethostname() != 'area51.cs.washington.edu' else os.path.join(self.data, 'index2word.pkl')
        self.wv_path = os.path.join(self.basepath, 'wv.model.vectors.npy') if socket.gethostname() != 'area51.cs.washington.edu' else os.path.join(self.data, 'wv.model.vectors.npy')
        self.log_folder = os.path.join(self.save_path, 'logs')

        #
        # self.model_path = os.path.join(self.basepath, 'wv.model')
        # # Not used in final mode, just kept for reference.
        # self.trainpicklepath = self.basepath + 'train.pkl'
        # self.evalpicklepath = self.basepath + 'val.pkl'
        # self.patient2hadmid_picklepath = os.path.join(self.basepath, 'patient2hadmid.pkl')
        # self.trainpicklepath_new = os.path.join(self.basepath , 'train_text_ts.pkl')
        # self.evalpicklepath_new = os.path.join(self.basepath , 'val_text_ts.pkl')
        # self.num_blocks = 3
        # self.mortality_class_ce_weigth = 10