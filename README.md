# HISEvent
This repository is a reproduction of the code and data of 'Hierarchical and Incremental Structural Entropy Minimization for Unsupervised Social Event Detection', accepted to AAAI 2024.

We observed that executing HISEvent on the entire Event2012 dataset required substantial memory resources, exceeding 62 GB of RAM. In contrast, FSD operates with significantly lower memory requirements (less than 32GB of RAM).

## To run HISEvent

### Step 1
Download the Event2012 and Event2018 datasets from [this Google drive link](https://drive.google.com/drive/folders/1i0VWPo4YeXYssVejOulDvgnYsBnAppYb?usp=drive_link).<br />
Place the entire ./raw_data folder under the root folder.

### Step 2
Preprocess datasets by running
```
python preprocess.py
```
### Step 3
Conduct social event detection by running
```
python run.py
```

