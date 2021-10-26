from torch.utils.data import Dataset
import numpy as np
from scipy import io
"""
Make a EEG dataset
X: EEG data
Y: KSS score
"""
class EEGDataset(Dataset):
    def __init__(self, dataset, subj_id):
        self.dataset = dataset
        self.len = len(dataset)
        self.subj_id = subj_id
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.dataset[idx][:,0:750][0:30].astype('float32')  # for only eeg
        y = self.dataset[idx][:,0][-1].astype('int64') # X 1개 (segment 1개)에 따라 y 1개

        X=np.expand_dims(X,axis=0) # (1, channel, time) batch 형태로
        # y=np.expand_dims(y,axis=0) # y는 단순히 숫자로만
    
        return X, y, self.subj_id

'''
dataset 생성
'''
class DriverDrowsiness_ReactionTime2():
    def __init__(self,root_path, SUBJECT_LIST):
        if root_path is None:
            raise ValueError('Data directory not specified!')

        self.datasets=[]
        self.subjectList=SUBJECT_LIST
            
        self.drowsy_num=0
        self.alert_num=0
        for idx, SBJ_NAME in enumerate(self.subjectList):
            ORI_DATA=io.loadmat(root_path+"s"+str(self.subjectList[idx])+'.mat') # data 불러오기
    
            self.x=ORI_DATA["epoch"]
            self.y=ORI_DATA["rt"][0]
            self.alertRT=ORI_DATA["tau0"]
            ##### 
            num_segment=[0,0]
            a_idx=[]
            d_idx=[]
            for i in range(self.x.shape[0]):
                rt=self.y[i]
                if rt<self.alertRT*1.5:
                    self.x[i][-1]=0
                    self.alert_num+=1
                    num_segment[0]+=1
                    a_idx.append(i)
                elif rt>self.alertRT*2.5:
                    self.x[i][-1]=1
                    self.drowsy_num+=1
                    num_segment[1]+=1
                    d_idx.append(i)
                
            print("s%d" % (self.subjectList[idx]), num_segment)
            # # nonsleep=sum(num_segment[0:6])
            # sleep=sum(num_segment[7:])
            # with open("subject_sleep_chancelevel.txt", 'a') as f:
            #     f.write("{}: sleep:{}\n".format(SBJ_NAME,100.*sleep/sum(num_segment)))

            wo_idx=a_idx+d_idx
            self.datasets.append(EEGDataset(self.x[wo_idx],idx))

        print("Total alert : drowsy =",self.alert_num,":",self.drowsy_num)

    def __getitem__(self, index):
        return self.datasets[index]# subject 1명씩

    def __len__(self):
        return len(self.datasets)