import DriverDrowsinessDataset_RT
import torch
import random
import numpy as np
import os
import argparse
from collections import Counter

from Model_DeepConvNet import DeepConvNet
from train_eval_predict import Train, valid_EVAL, predict_EVAL 

###########################################################################
# get weights (use the number of samples)
def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for y in dataset:
        c = y[1]
        counts[c] += 1 # count each class samples
        classes.append(c) 
    n_classes = len(counts)

    weight_per_class = {}
    for y in counts: # the key of counts
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights
###########################################################################

def Experiment(args, subjectList ,subject_id):
    # make a directory to save results, models
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
        os.makedirs(path + '/prediction')

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
    model = DeepConvNet(args.n_classes, args.n_channels, args.n_timewindow)
    if cuda: model.cuda(device=device) # connect DEVICE

    # OPTIMIZER, LEARNING SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-1)

    ########### train, valid data loader
    train_dataset=DriverDrowsinessDataset_RT.DriverDrowsiness_ReactionTime("train", args.data_root, subjectList) 
    train_set=train_dataset[subject_id]
    train_weights=make_weights_for_balanced_classes(train_set) # for class balanced sampling

    valid_dataset=DriverDrowsinessDataset_RT.DriverDrowsiness_ReactionTime("valid", args.data_root, subjectList) 
    valid_set=valid_dataset[subject_id]    

    train_loader = torch.utils.data.DataLoader(train_set, sampler=torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(train_weights)), batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size) #

    """ Training, Validation """
    best_balanced_acc=0
    for epochidx in range(1, args.epochs):
        print("EPOCH_IDX: ",epochidx)
        Train(20, model, device, train_loader, optimizer, scheduler) # train
        _, valid_score, valid_balanced_score = valid_EVAL(model, device, valid_loader) # valid
        
        # compare validation accuracy of this epoch with the best accuracy score
        # if validation accuracy >= best accuracy, then save model(.pt)
        if valid_balanced_score >= best_balanced_acc:
            print("Higher accuracy then before: epoch {}".format(epochidx))
            best_balanced_acc = valid_balanced_score
            torch.save(model.state_dict(), os.path.join(path, 'models',"subject{}_bestmodel".format(subject_id+1)))

    ############################## Inference ##############################
    test_dataset=DriverDrowsinessDataset_RT.DriverDrowsiness_ReactionTime("test", args.data_root, subjectList) # test sbj data
    test_set=test_dataset[subject_id]   
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    best_model = DeepConvNet(args.n_classes, args.n_channels, args.n_timewindow)
    best_model.load_state_dict(torch.load(os.path.join(path, 'models',"subject{}_bestmodel").format(subject_id+1), map_location=device))
    if cuda: 
        best_model.cuda(device=device)
    pred = predict_EVAL(best_model, device, test_loader)
    np.save(path + '/prediction/S'+str(subject_id)+"_y", pred)

    return valid_score, valid_balanced_score

###########################################################################
if __name__ == '__main__':

    """ Experiment Setting """ 
    # ARGUMENT
    parser = argparse.ArgumentParser(description='Reaction Time')
    parser.add_argument('--data-root', default='./dataset/')
    parser.add_argument('--save-root', default='./')
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--n_channels', default=30)
    parser.add_argument('--n_timewindow', default=750)
    parser.add_argument('--result-dir', default="MyModel" ) # save folder name
    parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training ')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')    
    args = parser.parse_args()
    
    args.gpuidx= 0
    exp_type=args.result_dir
    best_model_result_path=exp_type+'/'+exp_type+'_Accuracy.txt'

    subjectList=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] # subject number

    accs=[]
    balanced=[]
    for id in range(len(subjectList)):
        print("~"*25 + ' Valid Subject ' + str(subjectList[id]) + " " + "~"*25)
        acc, balancedacc = Experiment(args, subjectList, id)

        # the accuracy and loss earned from testing the best accuracy model from train and validation set
        accs.append(acc)
        balanced.append(balancedacc)

        print(f"TEST SUBJECT ID : {subjectList[id]}, ACCURACY : {acc:.2f}%, Balanced ACCURACY : {balancedacc:.2f}%")
        with open(best_model_result_path, 'a') as f:
            f.write("subject {}, acc: {}, balanced: {}\n".format(subjectList[id],acc,balancedacc)) # save test accuracy, test loss, sleep chance level

    print(f"TOTAL AVERAGE : {np.mean(accs):.2f}%, {np.mean(balanced):.2f}%")
    with open(best_model_result_path, 'a') as f:
            f.write("TOTAL AVERAGE: {} {}\n".format(np.mean(accs),np.mean(balanced))) # save mean test accuracy
