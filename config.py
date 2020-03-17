import os
import socket


class Config:
    """
    a class to store hyper parameters and filepath settings.
    """
    PROJECT_NAME = "ClinicalNotesICU"

    def __init__(self):
        # paths ##
        self.project_root = self.get_project_root()
        self.data_path = os.path.join(self.project_root, 'mimic3-benchmarks/data/')

        self.decompensation_path = os.path.join(self.data_path, 'decompensation')
        self.ihm_path = os.path.join(self.data_path, 'in-hospital-mortality')
        self.los_path = os.path.join(self.data_path, 'length-of-stay')

        self.norm_state = os.path.join(self.project_root, 'ihm_ts1.0.input_str-previous.start_time-zero.normalizer')
        self.text_path = os.path.join(self.project_root, 'text')
        self.embedding_path = '../BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.textdata_fixed = os.path.join(self.data_path, 'root', 'train_text_fixed')
        self.multitask_path = os.path.join(self.data_path, 'multitask')
        self.starttime_path = os.path.join(self.data_path, 'starttime.pkl')
        self.test_textdata_fixed = os.path.join(self.data_path, 'root', 'test_text_fixed')
        self.test_starttime_path = os.path.join(self.data_path, 'test_starttime.pkl')
        self.save_path = os.path.join(self.project_root, 'results')
        self.log_path = os.path.join(self.save_path, 'logs')

        self.index2word_path = os.path.join(self.project_root, 'index2word.pkl') if socket.gethostname() != 'area51.cs.washington.edu' else os.path.join(self.data, 'index2word.pkl')
        self.wv_path = os.path.join(self.project_root, 'wv.model.vectors.npy') if socket.gethostname() != 'area51.cs.washington.edu' else os.path.join(self.data, 'wv.model.vectors.npy')
        self.data = '../mimic3-benchmarks/data/' if socket.gethostname() != 'area51.cs.washington.edu' else r'/data/ClinicalNotesICU'

        # hyper parameters ##

        # lstm
        self.rnn_hidden_units = 256
        self.padding_type = 'Zero'
        self.buffer_size = 100
        self.timestep = 1.0
        self.dropout = 0.9

        # conv layer
        self.conv1d_channel_size = 256

        # misc
        self.maximum_number_events = 150
        self.imputation = 'previous'
        self.learning_rate = 1e-4
        self.break_text_at = 300
        self.small_part = False
        self.max_len = 1000
        self.num_epochs = 100
        self.batch_size = 5
        self.alpha = 0.9

        self.init_directories()

    def init_directories(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        return

    @staticmethod
    def get_project_root():
        path = os.getcwd()
        root_idx = path.find(Config.PROJECT_NAME)
        if root_idx == -1:
            raise NameError("Working directory outside of project folder. Could not establish project root path.")
        return path[0:root_idx+len(Config.PROJECT_NAME)]
