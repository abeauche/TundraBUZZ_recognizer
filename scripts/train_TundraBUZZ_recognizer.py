# The cnn module provides classes for training/predicting with various types of CNNs
from opensoundscape import CNN

# Other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess
from glob import glob
import sklearn
import os

# For preprocessor
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset
from opensoundscape import preprocess
import IPython.display as ipd
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid
from opensoundscape.ml.utils import collate_audio_samples_to_tensors
from opensoundscape.ml.dataloaders import SafeAudioDataloader

# Set up plotting
from matplotlib import pyplot as plt
from matplotlib import rcParams

#Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


plt.rcParams['figure.figsize']=[1335,5] #for large visuals

rcParams['figure.figsize'] = (10, 6)  # Example of setting figure size
rcParams['figure.dpi'] = 100           # Example of setting dpi

# Initiate wandb performance logging
import wandb
try:
   wandb.login()
   wandb_session = wandb.init(
       entity='abeauche-university-of-british-columbia', 
       project='Buzz Recognizer',
       name='LAST',
    )
except: #if wandb.init fails, don't use wandb logging
    print('failed to create wandb session. wandb session will be None')
    wandb_session = None

# Set seeds for reproducibility. Remove when running final model
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Make a list of all of the selection table files
raven_files = sorted(glob("./Data/TundraBUZZ_Annotations/*/*.txt"))
print("Raven files:", raven_files)

# Create a list of audio files, one corresponding to each Raven file 
# (Audio files have the same names as selection files with a different extension)
audio_files = sorted(glob("./Data/TundraBUZZ_RawAudio/*/*.wav") + glob("./Data/TundraBUZZ_RawAudio/*/*.mp3"))
print("Audio files:", audio_files)

from opensoundscape.annotations import BoxedAnnotations

# Create a dataframe of annotations
annotations = BoxedAnnotations.from_raven_files(
    raven_files, 
    "Annotation",
    audio_files, 
    keep_extra_columns=['Selection','View', 'Channel','Begin Time (s)', 
                        'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)','Annotation'])

# Access the underlying DataFrame
annotations_data = annotations.df
annotations_data.to_csv("./Data/Output_annotations/annotations_data.csv")

# Parameters to use for label creation
clip_duration = 0.3 # clip length (s)
clip_overlap = 0.15 # clip overlap (s)
min_label_overlap = 0.07 # 0.07 s of annotated call must be included to be considered "positive"
min_label_fraction = 0.8 # OR 80% of annotated call must be included
species_of_interest = ["BUZZ"] # matches annotation label

# Create dataframe of one-hot labels
clip_labels = annotations.clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
    min_label_fraction = min_label_fraction,
    class_subset = species_of_interest # You can comment this line out if you want to include all species.
)

# Access the underlying DataFrame
clip_labels.to_csv("./Data/Output_annotations/clip_labels.csv")

# Select all files from testing_data_raven as a test set
mask = clip_labels.reset_index()['file'].apply(lambda x: 'testing_data_audio' in x).values
test_set = clip_labels[mask]

# All other files will be used as a training/validation set
train_and_val_set = clip_labels.drop(test_set.index)

# Save .csv tables of the training/validation and test sets to keep a record of them
train_and_val_set.to_csv("./Data/Output_annotations/train_and_val_set.csv")
test_set.to_csv("./Data/Output_annotations/test_set.csv")

train_and_val_set = pd.read_csv('./Data/Output_annotations/train_and_val_set.csv',index_col=[0,1,2])
test_set = pd.read_csv('./Data/Output_annotations/test_set.csv',index_col=[0,1,2])


# Balance the negatives and positives so there are 2x the number of negatives as positives
from opensoundscape.data_selection import resample
# Identify positives (rows where any column is TRUE)
positives = train_and_val_set[train_and_val_set['BUZZ'] > 0]

# Identify negatives (rows where any column is FALSE)
negatives = train_and_val_set[train_and_val_set['BUZZ'] == 0]

# Check how many positives and negatives there are
num_positives = len(positives)
num_negatives = len(negatives)

# Sample the negatives to achieve twice as many negatives as positives
num_negatives_to_sample = min(2 * num_positives, num_negatives)

# Sample exactly `num_negatives_to_sample` negatives without replacement
negatives_downsampled = negatives.sample(num_negatives_to_sample, replace=False) # random state for reproducibility

# Concatenate positives and downsampled negatives into a balanced training set
balanced_train_and_val_set = pd.concat([positives, negatives_downsampled])

# Shuffle to randomize order of clips
from sklearn.utils import shuffle
balanced_train_and_val_set = shuffle(balanced_train_and_val_set) # random_state for reproducibility

# Split training data into training and validation sets
train_df, valid_df = sklearn.model_selection.train_test_split(balanced_train_and_val_set, test_size=0.2) # test_size = 0.2 to split data into 20% validation, random_state ensures reproducibility
train_df.to_csv("./Data/Output_annotations/train_set.csv")
valid_df.to_csv("./Data/Output_annotations/valid_set.csv")

# Modify the default preprocessor for training data **could add dB modifications
preprocessor = SpectrogramPreprocessor(sample_duration=0.3) # sample_duration must match clip_duration
train_dataset = AudioFileDataset(train_df, preprocessor)
preprocessor.pipeline
#preprocessor.pipeline.bandpass.set(min_f=0,max_f=4000) # eliminate unnecessary frequencies --> could be changed
preprocessor.pipeline.to_spec.params['overlap_fraction'] = 0.9 # fractional temporal overlap between consecutive windows
preprocessor.pipeline.to_spec.params.window_samples = 2048  # Example: Set window size to 2048 samples
preprocessor.pipeline.to_spec.params
print(preprocessor.pipeline.to_spec.params) # double check preprocessor parameters

# Visualize preprocessed tensors
tensors = [train_dataset[i].data for i in range(16,25)]
_ = show_tensor_grid(tensors,3)
#plt.show()

print("Reached model architecture step")

# Define model architecture
architecture = 'resnet34'

print("Reached model class step")

# Get class
class_list = list(train_df.columns)

print("Reached CNN creation step")

# Create CNN
model = CNN(
    architecture = architecture,
    classes = class_list,
    sample_duration = clip_duration # 3s, selected above
)


# Parallel computing to speed up training
# import torch
# if torch.backends.mps.is_available():
#     model.device='mps' #Apple Silicon
# elif torch.cuda.is_available():
#     model.device='cuda' #CUDA GPU  
# print(f'model.device is: {model.device}')

print("Reached model checkpoint step")

# Define checkpoint folder for saving training checkpoints
checkpoint_folder = Path("model_training_checkpoints")
try:
    checkpoint_folder.mkdir(exist_ok=True)
    print(f"Checkpoint folder created: {checkpoint_folder}")
except Exception as e:
    print(f"Failed to create checkpoint folder: {e}")

print("Reached model training step")

# Train model
if __name__ == '__main__':
    
    model.train(
    train_df,
    valid_df,
    epochs = 100, # based on when learning plateaus
    batch_size = 64,
    log_interval = 100, # log progress every 100 batches
    num_workers = 0, # parallelized cpu tasks for preprocessing
    wandb_session=wandb_session, 
    save_interval = 1, # save checkpoint every 10 epochs
    save_path = checkpoint_folder # location to save checkpoints
)

# Map CNN outputs
scores_df = model.predict(valid_df.head(),activation_layer='sigmoid')
