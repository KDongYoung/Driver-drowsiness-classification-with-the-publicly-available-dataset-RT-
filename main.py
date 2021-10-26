from torch.utils.tensorboard import SummaryWriter
import DriverDrowsinessDataset_RT
from torch.utils.data import random_split
import torch
import random
import numpy as np
import os
import argparse
from collections import Counter

# Model_DeepConvNet.py, train_eval_shallow.py
from Model_DeepConvNet import DeepConvNet
from train_eval_shallow import * # train, evaluation

###########################################################################
n_classes = 2  # rt class
n_channels=30 # eeg: 30
n_timewindow=750

if n_channels==30:
    exp_type="deepconvnet_rt"  

best_model_result_path=exp_type+'/'+exp_type+'_Accuracy.txt'

subjectList=[5, 15, 24, 31, 32, 36, 37, 38, 42, 46, 47, 48, 49, 52] # final
###########################################################################
# get weights (use the number of samples)
def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for y in dataset:
        y = int(y[1])
        counts[y] += 1 # count each class samples
        classes.append(y) 
    n_classes = len(counts)

    weight_per_class = {}
    for y in counts: # the key of counts
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def Experiment(subject_id):
    """
    Experiment Setting
    """ 
    # ARGUMENT
    parser = argparse.ArgumentParser(description='Reaction Time')
    parser.add_argument('--data-root', default='./pre_dataset/')
    parser.add_argument('--save-root', default='./') # ./default 
    parser.add_argument('--result-dir', default=exp_type) # save folder name
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.05)')                   
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    args = parser.parse_args()
    args.gpuidx= 0

    # make a directory to save results, models
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')

    # save ARGUEMENT
    with open(path + '/args.txt', 'w') as f:
        f.write(str(args))

    # connect GPU/CPU
    import torch.cuda
    cuda = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = 2020
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # MODEL
    res_layer=[]
    model = DeepConvNet(n_classes, n_channels, n_timewindow)

    if cuda:
        model.cuda(device=device) # connect DEVICE

    # OPTIMIZER, LEARNING SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-1)
    # tensorboard
    writer = SummaryWriter(f'{path}/{subjectList[id]}') # log directory

    """
    Balance the imbalance dataset by weighted random sampler (using weights) -> small dataset을 big dataset 갯수에 맞게 upsampling
    Dataset loader - train, valid, test
    train+valid : test = 8:2
    train : valid = 9:1
    """
    data=DriverDrowsinessDataset_RT.DriverDrowsiness_ReactionTime2(args.data_root, subjectList) # sbj의 데이터 불러오기
    dataset=data[subject_id]
    dataset_size=len(dataset)

    test_size=int(dataset_size*0.2)
    valid_size=int((dataset_size-test_size)*0.1)
    train_size=dataset_size-valid_size-test_size
    train_set, valid_set, test_set= random_split(dataset, [train_size, valid_size, test_size]) # train:valid=9:1, train+valid:test=8:2
    print(dataset_size, train_size, valid_size, test_size)
    train_weights=make_weights_for_balanced_classes(train_set)

    # train, vaidation, test loader
    train_loader = torch.utils.data.DataLoader(train_set, sampler=torch.utils.data.WeightedRandomSampler(train_weights,num_samples=len(train_weights)), batch_size=args.batch_size) # drop_last가 true일까 false일까?
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size) #
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    """
    Training, Validation
    """
    best_balanced_acc=0
    for epochidx in range(1, args.epochs):
        print("EPOCH_IDX: ",epochidx)
        train(20, model, device, train_loader, optimizer, scheduler) # train
        valid_loss, valid_score, valid_balanced_score = eval(model, device, valid_loader) # valid
        
        # compare validation accuracy of this epoch with the best accuracy score
        # if validation accuracy >= best accuracy, then save model(.pt)
        if valid_score >= best_balanced_acc:
            print("Higher accuracy then before: epoch {}".format(epochidx))
            best_balanced_acc = valid_balanced_score
            torch.save(model.state_dict(), os.path.join(path, 'models',"subject{}_bestmodel".format(subject_id+1)))
        
        writer.add_scalar('total/valid_loss', valid_loss, epochidx)
        writer.add_scalar('total/valid_acc', valid_score, epochidx)
        writer.add_scalar('total/valid_balanced_acc', valid_balanced_score, epochidx)    
    writer.close()

    # test the best accuracy model
    print("Testing...")
    best_model = DeepConvNet(n_classes, n_channels, n_timewindow)
    best_model.load_state_dict(torch.load(os.path.join(path, 'models',"subject{}_bestmodel").format(subject_id+1), map_location=device))
    if cuda: 
        best_model.cuda(device=device)
    test_loss, test_score, balanced_score = eval(best_model, device, test_loader)
    print("Best accuracy model => test_loss: {}, test_score: {}".format(test_loss,test_score))

    return test_score, balanced_score

###########################################################################
if __name__ == '__main__':
    accs=[]
    balanced=[]
    for id in range(len(subjectList)):
        print("~"*25 + ' Valid Subject ' + str(subjectList[id]) + " " + "~"*25)
        acc, balancedacc = Experiment(id)

        # the accuracy and loss earned from testing the best accuracy model from train and validation set
        accs.append(acc)
        balanced.append(balancedacc)

        print(f"TEST SUBJECT ID : {subjectList[id]}, ACCURACY : {acc:.2f}%, Balanced ACCURACY : {balancedacc:.2f}%")
        with open(best_model_result_path, 'a') as f:
            f.write("subject {}, acc: {}, balanced: {}\n".format(subjectList[id],acc,balancedacc)) # save test accuracy, test loss, sleep chance level

    print(f"TOTAL AVERAGE : {np.mean(accs):.2f}%, {np.mean(balanced):.2f}%")
    with open(best_model_result_path, 'a') as f:
            f.write("TOTAL AVERAGE: {} {}\n".format(np.mean(accs),np.mean(balanced))) # save mean test accuracy