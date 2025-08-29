# EEG Seizure Prediction
## Project Overview
This project is an epilepsy prediction system based on EEG (electroencephalogram) signals. Its main functions include reading and preprocessing EEG signals, feature extraction, dataset construction, model training and evaluation, etc. It aims to predict epileptic seizures by analyzing EEG signal features.
## Environment Dependencies :

python >= 3.12
torch             2.8.0+cu128
torchvision       0.23.0+cu128
scikit-learn      1.7.1
matplotlib        3.10.5
pandas            2.3.2
pyEDFlib          0.1.42
scipy             1.16.1
tqdm              4.67.1



## Main Directory

```
├── .git                    
├── .idea            
├── data                     
├── net                  
├── preprocessing                    # Pre-trained Data Processing
|  |──_pycache_
│  ├── edf_extraction.py         
│  ├── input.py           
│  ├── input_process_dataset.py              
│  ├── label_wrapper.py       
│  ├── make_dataset.py                  
│  ├── make_dataset_for_pred_v2.py            
│  ├── make_dataset_for_pred_v2_spectralgram.py
│  ├── make_dataset_for_pred_v2_zheer.py                  
│  ├── make_tfrecords.py         
│  ├── split_samples.py
│  ├── split_samples_for_pred.py                  
│  ├── split_samples_for_pred_v2.py         
│  ├── split_samples_for_pred_v2_zheer.py                                
├── train_pred                     
│  |──_pycache_            
│  ├── checkpoints         
│  ├── loss_compute.py       
│  ├── loss_compute_pred.py            
│  ├── loss_compute_torch.py           
│  ├── main_pred.py            
│  ├── main_pred_loocv.py                
│  ├── main_pred_loocv_v2.py      
│  ├── main_pred_test_loocv_v2.py            
│  ├── train_channel_frequency.py         
│  ├── train_pytorch.py               
│  ├── train_Transformer.py       
├── utils  
│  |──_pycache_            
│  ├── auc_utils.py         
│  ├── gcn_lstm_utils.py      
│  ├── gcn_utils.py           
│  ├── gcn_utils_pred.py           
│  ├── gcn_utils_torch.py           
│  ├── innovation_utils.py               
│  ├── length_GCN.py     
│  ├── length_HGCN.py           
│  ├── lstm_utils.py       
│  ├── manual_set_params.py             
│  ├── others.py            
│  ├── resnet_utils.py    
│  ├── set_transformer_utils.py           
│  ├── sharing_params.py                 
├── vene                   
├── mv-file.py
|──read_edf.py
|──temp.py   
|──test.py
```
## How to Use
1. Open the project's GitHub repository page (https://github.com/seizure-anonymous/Seizure-prediction.git).
2. Click the Code button in the upper right corner of the page and select a download method:
 **Clone via Git **：
     ```bash
     git clone https://github.com/seizure-anonymous/Seizure-prediction.git
3. Make sure you have conda installed and create / activate a dedicated virtual environment(recommended):
    ```bash
    conda create -n eeg python=3.12 -y
    conda activate eeg
4. To install the required dependencies, run the following command:
    ```bash
    pip install numpy
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
    pip install matplotlib
    pip install tqdm
    pip install scipy
    conda install -c conda-forge pyedflib 
    #When pip builds pyEDFlib’s Cython extension locally, it reads the source file pyedflib/_extensions/_pyedflib.pyx with the system default encoding (GBK on Chinese Windows).Because the file contains non-GBK characters, this triggers a UnicodeDecodeError.Activate your virtual environment and simply install the pre-built conda-forge package instead—conda-forge already provides Windows wheels/conda packages—so no local compilation is needed and the encoding error is avoided.
    pip install scikit-learn
    pip install pandas
5. Run the split_samples_for_pred_v2.py  to get segmented EEG signals.
    ```bash
    cd preprocessing
    python split_samples_for_pred_v2.py
6. Run make_dataset_for_pred_v2.py to get the features and labels of EEG signals(It will may take a long time).
    ```bash
    python make_dataset_for_pred_v2.py
7. As of Step 6, the data-preprocessing phase has been completed.
## Key Function Specification


**split_samples_for_pred.py**: This script performs the ‘segmentation’ step: it slices each subject’s CHB-MIT long-term EEG into numerous fixed-length (4 s) clips and produces a corresponding list file of preictal / interictal labels, which make_dataset_for_pred_v2.py will later use for feature extraction.
| core function | Description |
|---|---|
| LabelWrapper | Parse chbXX-summary.txt to extract the onset & offset times of all seizures. |
| check_file_channel | Ensure every .edf contains all channels specified in params.normal_signal; discard any file that does not. |
| build_intervals | Generate the raw preictal / interictal intervals (large contiguous blocks) for each subject. |
| build_split | Slide a 4-second window over the already-defined preictal and interictal blocks to cut them into small 4-s segments, and write the index of each segment into the corresponding *.txt file. |


**make_dataset_for_pred_v2.py**:Iterate over each 4-second split listed in the generated *.txt files, extract PSD + ratio features for every segment, and save the corresponding labels and feature tensors as label and feature files for immediate use in training or validation.
| core function | Description |
|---|---|
| ExtractSignal | On-demand CHB-MIT EDF opener that caches the requested channels’ full-length signals and instantly returns any 4-second segment’s raw waveform plus its PSD+ratio features. |
| run_extract(sample, label, visual, extract) | For a given 4-second segment tuple (file, start_sec, end_sec), invoke ExtractSignal to extract the raw multi-channel waveform and immediately compute its PSD + ratio features. |
| build_tfrecords_pred(path, file_name) | Read the list of (edf, start, end) tuples from file_name (*.txt).Call run_extract() on each tuple to obtain its features.Append the corresponding label and feature tensor to data_label and data_feature, respectively.Finally, save the two lists as path/{label}/label.pt and path/{label}/feature.pt with torch.save(). |
