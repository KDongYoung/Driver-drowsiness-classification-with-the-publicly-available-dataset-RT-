import numpy as np
from scipy import io

dataset_path = "원본 데이터 경로"  # 원본 데이터 경로 설정
pre_dataset_path = "전처리 후 데이터 경로"  # 전처리 후 데이터 경로 설정
freq=500
second=3

def load_data(path,sbj_num):
    FILE = io.loadmat(path + 's%d.mat' % (sbj_num)) # 데이터 불러오기
    name = FILE["dataset"]['name'][0][0][0]
    eeg_data = FILE["dataset"]['data'][0][0]
    event = FILE["dataset"]['event'][0][0][0] 

    return name, eeg_data, event

def downsampling(df):
    down=[]
    for i in range(len(df)):
        down.append(list(df[i][::2]))
    return np.array(down)

def segment_start_end_row(range_start,range_end):
    a=list(range(range_start,range_end+2,2))
    b=(a[-1]-a[0])//(second*freq/2) # downsampling
    ses=[]
    for i in range(int(b)):
        se=[]
        se.append(int(a[-1]-(second*freq/2)*(i+1))) # start
        se.append(int(a[-1]-(second*freq/2)*i)) # end  => start:end
        ses.append(se)
    return ses,int(b)

def epoching(df, seg_se, tau0):
    total_di=[]
    total_rt=[]
    total_df=np.array([])
    for k in range(len(seg_se)):
        seg_df=df[:,seg_se[k][0]:seg_se[k][1]]

        di=[]
        for i in range(len(seg_df[-1])):
            di.append(max(0,(1-np.exp(-(seg_df[-1][i]-tau0)))/(1+np.exp(-(seg_df[-1][i]-tau0)))))
      
        total_di.append(np.mean(di)) # 구간의 평균 값을 대표 di로 지정
        total_rt.append(np.mean(seg_df[-1])) # 구간의 평균 값을 대표 rt로 지정
        di=np.expand_dims(np.array(di),axis=0)
        seg_df=np.concatenate((seg_df,di))
        seg_df=np.expand_dims(seg_df,axis=0)

        if k==0:
            total_df=seg_df
        else:
            total_df=np.concatenate((total_df,seg_df))
            
    return total_df, total_di, total_rt

def main(subject_id):
    _, eeg_data, event=load_data(dataset_path,subject_id)
    exp_duration=len(eeg_data[0])
    
    event_type=event[:]["type"].tolist()
    event_latency=event[:]["latency"].tolist()
    event_init_time=event[:]["init_time"].tolist()
    
    t=[]
    num_type=[0,0,0,0] # 251, 252, 253, 254
    row12=[] # deviation onset
    row3=[] # response onset
    row4=[] # response offset
    for i in range(len(event_type)):
        if event_type[i][0][0]==251:
            num_type[0]+=1
            row12.append(event_latency[i][0][0])
        elif event_type[i][0][0]==252:
            num_type[1]+=1
            row12.append(event_latency[i][0][0])
        elif event_type[i][0][0]==253:
            num_type[2]+=1
            row3.append(event_latency[i][0][0])
            t.append(event_init_time[i][0][0]-event_init_time[i-1][0][0])
        elif event_type[i][0][0]==254:
            num_type[3]+=1
            row4.append(event_latency[i][0][0])
    # print(num_type)

    # row3 - row12 => reaction time
    rt=[]
    rt_zip = zip(row12, row3)
    for r12, r3 in rt_zip:
        rt.append((r3-r12)/500) # time은 /500 (sampling rate 500Hz)
    # print(rt)
    
    interRT=[-1]*exp_duration
    interRT[0]=np.mean(rt[:round(len(rt)/2)]) # 시작 후 중간까지 시간의 평균
    interRT[-1]=np.mean(rt[round(len(rt)/2):]) # 나머지 절반 시간의 평균
    seg_se=[]
    for i in range(len(rt)+1):
        if i==0:
            # seg_num.append((row12[0]-1)//(second*freq))
            interRT[0:row12[i]-1]=np.linspace(interRT[0],rt[0],row12[0]-1)
            # seg=segment_start_end_row(0,row12[0]-1,row12[0]-1)
            # seg_se.extend(seg)
        elif i==len(rt):
            # seg_num.append((exp_duration-row4[i-1])//(second*freq))
            interRT[row4[i-1]:]=np.linspace(rt[i-1],interRT[-1],exp_duration-row4[i-1])
            # seg=segment_start_end_row(row4[i-1],exp_duration-1,exp_duration-row4[i-1])
            # seg_se.extend(seg)
        else:
            # seg_num.append((row12[i]-row4[i-1]-1)//(second*freq))
            interRT[row4[i-1]:row12[i]-1]=np.linspace(rt[i-1],rt[i],row12[i]-row4[i-1]-1)
            # seg=segment_start_end_row(row4[i-1],row12[i]-1,row12[i]-row4[i-1]-1)
            # seg_se.extend(seg)
    # print(subject_id,seg_se)
    interRT_df=np.array([interRT])
    total_df=np.concatenate((eeg_data,interRT_df))

    # downsampling
    d_total=downsampling(total_df)
    ld_index=list(np.where(d_total[-1]==-1)[0])
    total_df_prepro=np.delete(d_total[:],ld_index,axis=1)

    seg_se=[]
    nums=0
    for i in range(len(ld_index)+1):
        if i!=0 and i!=len(ld_index):
            if ld_index[i]-ld_index[i-1]>1:
                seg,num=segment_start_end_row(ld_index[i-1]+1,ld_index[i]-1)
                seg_se.extend(seg)
                nums+=num
        elif i==len(ld_index):
            seg,num=segment_start_end_row(ld_index[-1]+1,len(d_total[0])-1)
            seg_se.extend(seg)
            nums+=num
        elif i==0:
            seg,num=segment_start_end_row(0,ld_index[0]-1)
            seg_se.extend(seg)
            nums+=num
    # print(seg_se)
    print(subject_id, len(seg_se))
      
    # drowsiness index로 변경 
    tau0=np.percentile(np.array(total_df_prepro[-1]),5)
    print(tau0*1.5, tau0*2.5, np.min(rt), np.max(rt), np.mean(rt), np.median(rt))
    # epoch하고 그 다음에 di 계산하자
    total_epoch, total_di, total_rt = epoching(d_total,seg_se,tau0) # 모든 epoch들 저장하기
    
    total={"epoch":total_epoch, "rt": total_rt, "di":total_di, "tau0": tau0}
    
    io.savemat(pre_dataset_path+'s'+str(subject_id)+'.mat', total, oned_as='row')



if __name__ == '__main__': # for 문으로 모든 sbj에 접근
    for sbj_id in range(60): # 62 subject
        main(sbj_id+1)

    # sbj_num=[11,12,29,33,42,44,45,46,47,49,50]
