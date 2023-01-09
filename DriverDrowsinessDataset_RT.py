from torch.utils.data import Dataset
import numpy as np

"""
Make a EEG dataset
X: EEG data
Y: KSS score
"""
class EEGDataset(Dataset):
    def __init__(self, phase, DATA, subj_id):
        self.phase=phase
        self.DATA = DATA
        self.len = len(DATA)
        self.subj_id = subj_id
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        # EEG 30 channel + Reaction Time
        if self.phase !="test":
            X = self.DATA[idx][0][0] # for only eeg
            y = self.DATA[idx][1] 
            X=np.expand_dims(X,axis=0) # (1, channel, time) batch shape

            return X, y, self.subj_id
        else:
            X = self.DATA[idx]  # for only eeg
            return X

''' dataset 생성 '''
class DriverDrowsiness_ReactionTime():
    def __init__(self, phase, root_path, SUBJECT_LIST):
        if root_path is None:
            raise ValueError('Data directory not specified!')

        self.datasets=[]
        self.subjectList=SUBJECT_LIST
        
        if phase != "test":
            for idx, SBJ_NAME in enumerate(self.subjectList):
                ORI_DATA=np.load(root_path+phase+"/S"+str(SBJ_NAME)+"_"+phase+".npy", allow_pickle=True) # load train, valid data 
                
                DATA=[]
                for i in range(len(ORI_DATA)//3):
                    datas=[ORI_DATA[i*3],ORI_DATA[i*3+1],ORI_DATA[i*3+2]]
                    DATA.append(datas)

                self.datasets.append(EEGDataset(phase, DATA, idx))
                print("s"+str(SBJ_NAME)+" segment number:",len(ORI_DATA))
        else: 
            for idx, SBJ_NAME in enumerate(self.subjectList):
                ORI_DATA=np.load(root_path+"/test_x/S"+str(SBJ_NAME)+"_x.npy", allow_pickle=True) # load test data 

                DATA=[]
                for i in range(len(ORI_DATA)):
                    datas=[ORI_DATA[i]]
                    DATA.append(datas)

                self.datasets.append(EEGDataset(phase, DATA, idx))
                print("s"+str(SBJ_NAME)+" segment number:",len(ORI_DATA))

    def __getitem__(self, index):
        return self.datasets[index] # one subject at a time 
    
    def __len__(self):
        return len(self.datasets)
