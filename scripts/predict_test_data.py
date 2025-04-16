# Make predictions using pika recognizer

from opensoundscape.ml.cnn import load_model
from opensoundscape import Audio
import opensoundscape

# Other utilities and packages
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import subprocess
from os import listdir
#set up plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals

#load model
model = load_model('E:\\Bumblebee_Recognizer\\model_training_checkpoints_2\\epoch-25.model') #set this to where you've saved the model




#load audio files 
audio_files_test = sorted(glob("E:\\Bumblebee_Recognizer\\Data\\TundraBUZZ_RawAudio\\testing_data\\*.wav")) #set this to where the audio files are

#Inspect the audio_files list to ensure you are processing the correct data
print(audio_files_test)
#res = [f for f in audio_files if "20240702" in f] 
#res = [f for f in audio_files if "20240725" in f] 
#res2 =  [f for f in res if  "MD_26Jul2024"  in f]

if __name__ == '__main__':
    scores = model.predict(audio_files_test, clip_overlap = 0.15, num_workers = 20, batch_size=36)
    scores.head()
    scores.to_csv("E:\\Bumblebee_Recognizer\\Data\\TundraBUZZ_RawAudio\\testing_data\\predict_score_test.csv") #set this to where you want to save the predictions


