# Driver drowsiness classification with the publicly available dataset

This is an proposed midterm project for "Applications and practice in neural networks" course

Classifiying driver drowsiness based on the dataset from the dataset available 
[\[Original version\]](https://figshare.com/articles/dataset/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task/6427334/5)
[\[Preprocessed version\]](https://figshare.com/articles/dataset/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task_preprocessed_dataset_/7666055/3)

1. S%d_train.npy, S%d_valid.npy file includes the EEG data and class label (alert:0, drowsy:1)

### Dataset Citation

```
@article{cao2019multi,
  title={Multi-channel EEG recordings during a sustained-attention driving task},
  author={Cao, Zehong and Chuang, Chun-Hsiang and King, Jung-Kai and Lin, Chin-Teng},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={1--8},
  year={2019},
  publisher={Nature Publishing Group}
}
```


## Rquirements

python ≥ 3.8.10   numpy ≥ 1.20.3  pandas ≥ 1.2.5  scipy ≥ 1.6.2   torch ≥ 1.9.0

sklearn   random  os  argparse  collection

IF using a Docker, use the recent image file ("pytorch:22.04-py3") uploaded in the [\[NVIDIA pytorch\]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) when running a container

-------------

If you have further questions of the project, please contact dy_kim@korea.ac.kr
