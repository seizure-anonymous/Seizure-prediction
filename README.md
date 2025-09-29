# EEG Seizure Prediction
## Project Overview
This project is an epilepsy prediction system based on EEG (electroencephalogram) signals. Its main functions include reading and preprocessing EEG signals, feature extraction (such as PSD, time-domain features, etc.), dataset construction, model training and evaluation, etc. It aims to predict epileptic seizures by analyzing EEG signal features.
## Environment Dependencies :
* system Windows10
* python >= 3.12
* torch             2.8.0+cu128
* torchvision       0.23.0+cu128
* scikit-learn      1.7.1
* matplotlib        3.10.5
* pandas            2.3.2
* pyEDFlib          0.1.42
* scipy             1.16.1
* tqdm              4.67.1

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
5. Locate the ```sharing_params.py``` file in the ```utils folder```, then modify the path of dataset_dir in line 99 so that this path can find   the chbmit-eeg data to be processed on your computer.
6. Run the ```split_samples_for_pred_v2.py```  to get segmented EEG signals.
    ```bash
    cd preprocessing
    python split_samples_for_pred_v2.py
7. Run ```make_dataset_for_pred_v2.py``` to get the features and labels of EEG signals(It will may take a long time).
    ```bash
    python make_dataset_for_pred_v2.py
8. As of Step 7, the data-preprocessing phase has been completed.
9. Run ```main_pred_loocv_v2.py``` to train a model and predict EEG signal patterns associated with epilepsy
   ```bash
   cd ..
   cd train_pred
   python main_pred_loocv_v2.py --exp_name Trans_BF_0p5H_1H_not_event_other_right --model All_Transformer --patient 12 --seizure_test 5 --seizure_val 5 --channel 8,9,16
   ```
   You can modify the command-line arguments to change the specified model type, patient ID, and other parameters.
10. You have completed the deployment of the model,have fun!
## Key Function Specification

**split_samples_for_pred.py**: This script performs the ‘segmentation’ step: it slices each subject’s CHB-MIT long-term EEG into numerous fixed-length (4 s) clips and produces a corresponding list file of preictal / interictal labels, which make_dataset_for_pred_v2.py will later use for feature extraction.
| core function | Description |
|---|---|
| LabelWrapper | Parse chbXX-summary.txt to extract the onset & offset times of all seizures. |
| check_file_channel(self, file) | Ensure every .edf contains all channels specified in params.normal_signal; discard any file that does not. |
| build_intervals(self) | Generate the raw preictal / interictal intervals (large contiguous blocks) for each subject. |
| build_split(self) | Slide a 4-second window over the already-defined preictal and interictal blocks to cut them into small 4-s segments, and write the index of each segment into the corresponding *.txt file. |

**make_dataset_for_pred_v2.py**:Iterate over each 4-second split listed in the generated *.txt files, extract PSD + ratio features for every segment, and save the corresponding labels and feature tensors as label and feature files for immediate use in training or validation.
| core function | Description |
|---|---|
| ExtractSignal(object) | On-demand CHB-MIT EDF opener that caches the requested channels’ full-length signals and instantly returns any 4-second segment’s raw waveform plus its PSD+ratio features. |
| run_extract(sample, label, visual, extract) | For a given 4-second segment tuple (file, start_sec, end_sec), invoke ExtractSignal to extract the raw multi-channel waveform and immediately compute its PSD + ratio features. |
| build_tfrecords_pred(path, file_name) | Read the list of (edf, start, end) tuples from file_name (*.txt).Call run_extract() on each tuple to obtain its features.Append the corresponding label and feature tensor to data_label and data_feature, respectively.Finally, save the two lists as path/{label}/label.pt and path/{label}/feature.pt with torch.save(). |

**main_pred_loocv_v2.py**:Iterate over each 4-second split listed in the generated *.txt files, extract PSD + ratio features for every segment, and save the corresponding labels and feature tensors as label and feature files for immediate use in training or validation.
| core function | Description |
|---|---|
| __init__() | Create ```checkpoints```-related directories to save models and logs |
train(args, io) 

                 1.Load the corresponding model (such as GcnNet, All_Transformer, etc.) according to args.model and use multi - GPU training.
                 2.Load the training set and validation set data, and use DataLoader for processing.
                 3.Define the optimizer (SGD or Adam) and the learning rate scheduler (ReduceLROnPlateau).
                 4.Training loop: Perform forward propagation to calculate the loss, conduct backpropagation to update parameters, 
                 and  record training 5.5.indicators.
                 5.Validation process: Calculate the accuracy of the validation set and save the best model.

test(args, io) 

                 1.Load the best trained model (selected based on accuracy).
                 2.Load the test set data.
                 3.Perform model inference and calculate metrics such as loss and accuracy on the test set.
                 4.Record the prediction results to a file, including statistics of continuous positive predictions and other information.
                                                              
## Parameter Setting


| Parameter Name         | Default Value | Core Function                 | Supplementary Explanation                                                                 |
|------------------------|---------------|--------------------------------|------------------------------------------------------------------------------------------|
| `--exp_name`           | `exp`         | Define experiment name         | Used to create an exclusive result directory (e.g., `checkpoints/exp_name`), distinguishing models, logs, and prediction results of different experiments for easy management and reproduction |
| `--model`              | `HGcnNet`     | Specify the type of model to use| Supports models such as `All_Transformer`, `LSTM_Transformer`, and `CNN_Transformer`; select the target model for training/testing via this parameter |
| `--model_path`         | `""`          | Specify pre-trained model path | If you need to continue training based on an existing model or test directly, pass the path of the pre-trained model file  to avoid training from scratch |
| `--use_sgd`            | `True`        | Select optimizer type          | Set to `True` to use **SGD optimizer**, and `False` to use **Adam optimizer**; controls the parameter update algorithm |
| `--momentum`           | `0.9`         | Set SGD momentum               | Takes effect only when `--use_sgd=True`; accelerates SGD convergence, reduces oscillations, and the common value is 0.9 |
| `--eval`               | `False`       | Control whether to perform only evaluation | Set to `True` to skip training and directly load the model for testing; set to `False` to execute the complete "training → testing" process |
| `--no_cuda`            | `False`       | Force to use CPU for computation | Set to `True` to use CPU regardless of whether a GPU is available; set to `False` to automatically use available GPU for acceleration |
| `--seed`               | `1`           | Fix random seed                | Ensures reproducible results of random operations such as data shuffling and model initialization, avoiding fluctuations in experimental results due to randomness |
| `--batch_size`         | `16`          | Set batch sample size          | The number of samples processed by the model at one time during training/testing, which affects training efficiency (larger batch size means higher efficiency) and memory usage |
| `--epochs`             | `30`          | Set total training epochs      | The number of times the training set is traversed by the model; more epochs mean more sufficient learning, but may increase the risk of overfitting and training time |
| `--lr`                 | `0.1`         | Set initial learning rate      | Controls the step size of parameter updates: too large a step size may cause divergence, while too small a step size leads to slow convergence; common value is 0.1 for SGD and 0.001 for Adam |
| `--dropout`            | `0.5`         | Set Dropout retention probability | The probability that neurons are "not discarded" during training (e.g., 0.5 means 50% of neurons are randomly discarded), used to prevent overfitting |
| `--MOVING_AVERAGE_DECAY` | `0.999`      | Set parameter moving average decay rate | Used for moving average update of model parameters; a larger decay rate means a stronger impact of historical parameters on the current update, improving model stability |
| `--use_batch_norm`     | `1`           | Control whether to use Batch Normalization (BN) | Set to `1` to enable BN layer, which accelerates training and alleviates gradient disappearance; set to `0` to disable BN layer |
| `--restore_step`       | `0`           | Specify the step to resume training | When training is interrupted, pass the interruption step to resume training from that point without starting over |
| `--train_set`          | `""`          | Specify training set information | Reserved parameter for identifying the path or configuration of the training set (not fully enabled in the current script, can be extended as needed) |
| `--eval_set`           | `""`          | Specify validation set information | Reserved parameter for identifying the path or configuration of the validation set (not fully enabled in the current script, can be extended as needed) |
| `--test_set`           | `""`          | Specify test set information    | Reserved parameter for identifying the path or configuration of the test set (not fully enabled in the current script, can be extended as needed) |
| `--independent`        | `False`       | Control whether it is an "individual-independent" experiment | Set to `True` to model independently by individual (e.g., training separately for a single patient); set to `False` to train jointly with multi-individual data |
| `--overlap`            | `True`        | Control whether EEG data segments overlap | When processing EEG time series data, set to `True` to allow overlapping sampling of data segments (improving data utilization), and `False` to disable overlapping |
| `--loocv`              | `False`       | Control whether to use Leave-One-Out Cross-Validation (LOOCV) | Set to `True` to use LOOCV for model evaluation (e.g., "leave one patient's data as the test set"); set to `False` to use ordinary cross-validation |
| `--patient`            | `False`       | Specify target patient ID       | For the multi-patient CHB-MIT dataset, select data of a specific patient (e.g., `--patient 12` means using EEG data of patient 12) |
| `--seizure_test`       | `None`        | Specify the seizure number for testing | Select specific seizure data of a patient as the test set (e.g., `--seizure_test 5` means using the 5th seizure data of the patient for testing) |
| `--seizure_val`        | `None`        | Specify the seizure number for validation | Select specific seizure data of a patient as the validation set (e.g., `--seizure_val 5` means using the 5th seizure data of the patient for validation) |
| `--channel`            | `empty`       | Specify EEG channels to use    | Select specific EEG channels (e.g., `--channel 8,9,16` means using channels 8, 9, and 16); set to `empty` to use all channels |
