# EEG Seizure Prediction
## Project Overview
This project is an epilepsy prediction system based on EEG (electroencephalogram) signals. Its main functions include reading and preprocessing EEG signals, feature extraction, model training and evaluation. It aims to predict epileptic seizures by analyzing EEG signal features.
## Environment Dependencies :
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
1. Locate the `sharing_params.py` file in the `utils folder`, then modify the path of dataset\_dir in line 99 so that this path can find   the chbmit-eeg data to be processed on your computer.
2. Run the `split_samples_for_pred_v2.py`  to get segmented EEG signals.
   ```bash
   cd preprocessing
   python split_samples_for_pred_v2.py
   ```
3. Run `make_dataset_for_pred_v2.py` to get the features and labels of EEG signals(It may take some time).
   ```bash
   python make_dataset_for_pred_v2.py
   ```
4. After Step 7, the data-preprocessing phase has been completed.
5. Run `main_pred_loocv_v2.py` to train a model and predict EEG signal patterns associated with epilepsy
   ```bash
   cd train_pred
   python main_pred_loocv_v2.py --exp_name Trans_BF_0p5H_1H_not_event_other_right --model All_Transformer --patient 12 --seizure_test 5 --seizure_val 5 --channel 8,9,16
   ```
   You can modify the command-line arguments to change the specified model type, patient ID, and other parameters.

   ## Parameter Setting

   | Parameter Name           | Default Value | Core Function                                                 | Supplementary Explanation                                                                                                                                                                      |
   | :----------------------- | :------------ | :------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `--exp_name`             | `exp`         | Define experiment name                                        | Used to create an exclusive result directory (e.g., `checkpoints/exp_name`), distinguishing models, logs, and prediction results of different experiments for easy management and reproduction |
   | `--model`                | `All_Transformer`     | Specify the type of model to use                              | Supports models such as `All_Transformer`, `LSTM_Transformer`, and `CNN_Transformer`; select the target model for training/testing via this parameter                                          |
   | `--model_path`           | `""`          | Specify pre-trained model path                                | If you need to continue training based on an existing model or test directly, pass the path of the pre-trained model file to avoid training from scratch                                       |
   | `--use_sgd`              | `True`        | Select optimizer type                                         |                                                                     |
   | `--momentum`             | `0.9`         | Set SGD momentum                                              |                                                                   |
   | `--eval`                 | `False`       | Control whether to perform only evaluation                    | Set to `True` to skip training and directly load the model for testing; set to `False` to execute the complete "training → testing" process                                                    |
   | `--no_cuda`              | `False`       | Force to use CPU for computation                              | Set to `True` to use CPU regardless of whether a GPU is available; set to `False` to automatically use available GPU for acceleration                                                          |
   | `--seed`                 | `1`           | Fix random seed                                               | Ensures reproducible results of random operations such as data shuffling and model initialization, avoiding fluctuations in experimental results due to randomness                             |
   | `--batch_size`           | `16`          | Set batch sample size                                         | The number of samples processed by the model at one time during training/testing, which affects training efficiency (larger batch size means higher efficiency) and memory usage               |
   | `--epochs`               | `30`          | Set total training epochs                                     | The number of times the training set is traversed by the model; more epochs mean more sufficient learning, but may increase the risk of overfitting and training time                          |
   | `--lr`                   | `0.1`         | Set initial learning rate                                     |
   | `--dropout`              | `0.5`         | Set Dropout retention probability                             | The probability that neurons are "not discarded" during training (e.g., 0.5 means 50% of neurons are randomly discarded), used to prevent overfitting                                          |
   | `--MOVING_AVERAGE_DECAY` | `0.999`       | Set parameter moving average decay rate                       | Used for moving average update of model parameters; a larger decay rate means a stronger impact of historical parameters on the current update, improving model stability                      |
   | `--use_batch_norm`       | `1`           | Control whether to use Batch Normalization (BN)               |                                                               |
   | `--restore_step`         | `0`           | Specify the step to resume training                           | When training is interrupted, pass the interruption step to resume training from that point without starting over                                                                              |
   | `--train_set`            | `""`          | Specify training set information                              | Reserved parameter for identifying the path or configuration of the training set (not fully enabled in the current script, can be extended as needed)                                          |
   | `--eval_set`             | `""`          | Specify validation set information                            | Reserved parameter for identifying the path or configuration of the validation set (not fully enabled in the current script, can be extended as needed)                                        |
   | `--test_set`             | `""`          | Specify test set information                                  | Reserved parameter for identifying the path or configuration of the test set (not fully enabled in the current script, can be extended as needed)                                              |
   | `--independent`          | `False`       | Control whether it is an "individual-independent" experiment  |                                                                                                                                                                                                |
   | `--overlap`              | `True`        | Control whether EEG data segments overlap                     |                                                                                                                                                                                                |
   | `--loocv`                | `False`       | Control whether to use Leave-One-Out Cross-Validation (LOOCV) |                                                                                                                                                                                                |
   | `--patient`              | `False`       | Specify target patient ID                                     | For the multi-patient CHB-MIT dataset, select data of a specific patient (e.g., `--patient 12` means using EEG data of patient 12)                                                             |
   | `--seizure_test`         | `None`        | Specify the seizure number for testing                        | Select specific seizure data of a patient as the test set (e.g., `--seizure_test 5` means using the 5th seizure data of the patient for testing)                                               |
   | `--seizure_val`          | `None`        | Specify the seizure number for validation                     | Select specific seizure data of a patient as the validation set (e.g., `--seizure_val 5` means using the 5th seizure data of the patient for validation)                                       |
   | `--channel`              | `empty`       | Specify EEG channels to use                                   | Select specific EEG channels (e.g., `--channel 8,9,16` means using channels 8, 9, and 16); set to `empty` to use all channels                                                                  |

## Key Function Specification

**split\_samples\_for\_pred.py**: This script performs the ‘segmentation’ step: it slices each subject’s CHB-MIT long-term EEG into numerous fixed-length (4 s) clips and produces a corresponding list file of preictal / interictal labels, which make\_dataset\_for\_pred\_v2.py will later use for feature extraction.

**make\_dataset\_for\_pred\_v2.py**:Iterate over each 4-second split listed in the generated \*.txt files, extract PSD + ratio features for every segment, and save the corresponding labels and feature tensors as label and feature files for immediate use in training or validation.

**main\_pred\_loocv\_v2.py**:Iterate over each 4-second split listed in the generated \*.txt files, extract PSD + ratio features for every segment, and save the corresponding labels and feature tensors as label and feature files for immediate use in training or validation.          &#x20;

##
