clear all; close all; clc;
fileName={'s01_051017m', 's01_060227n', 's01_060926_1n', 's01_060926_2n', ...
    's01_061102n', 's02_050921m', 's02_051115m', 's04_051130m', 's05_051120m', 's05_060308n', ...
    's05_061019m', 's05_061101n', 's06_051119m', 's09_060313n', 's09_060317n', 's09_060720_1n', ...
    's11_060920_1n', 's12_060710_1m', 's12_060710_2m', 's13_060213m', 's13_060217m', 's14_060319m', ...
    's14_060319n', 's22_080513m', 's22_090825n', 's22_090922m', 's22_091006m', 's23_060711_1m', ...
    's31_061020m', 's31_061103n', 's35_070115m', 's35_070322n', 's40_070124n', 's40_070131m', ...
    's41_061225n', 's41_080520m', 's41_080530n', 's41_090813m', 's41_091104n', 's42_061229n', ...
    's42_070105n', 's43_070202m', 's43_070205n', 's43_070208n', 's44_070126m', 's44_070205n', ...
    's44_070209m', 's44_070325n', 's45_070307n', 's45_070321n', 's48_080501n', 's49_080522n', ...
    's49_080527n', 's49_080602m', 's50_080725n', 's50_080731m', 's52_081017n', 's53_081018n', ...
    's53_090918n', 's53_090925m', 's54_081226m', 's55_090930n'};

saveRT = 'C:\Users\Dong Young\Desktop\ReactionTime';
% file = 'D:\Reaction Time\Preprocessed_Total';
file2='D:\Reaction Time\Preprocessed';

for s = 1 : length(fileName)
    disp(fileName{s});
    
    filename=cellstr(fileName{s}+".set");
    path=strcat(file2,'\',fileName{s});
   
    RT_DATA=pop_loadset(filename,path);
%     load('-mat',file+fileName{s}+".set")
        
    dataset.data=RT_DATA.data;
    dataset.event=RT_DATA.event;
    dataset.name=RT_DATA.setname;
    dataset.freq=RT_DATA.srate;
    
    save([saveRT, '\dataset\s', num2str(s)],"dataset")
end