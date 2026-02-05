# Make predictions using TundraBUZZ recognizer

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
model = load_model("/Users/alexandrebeauchemin/TundraBUZZ_recognizer_github/model_training_checkpoints/epoch-25.model") #set to where model is saved


#load audio files
audio_files = sorted(glob("/Volumes/TundraBUZZ/data/raw/aru_audio/ARUQ10_2025_test/*.wav")) #set to where the audio files are

#Inspect the audio_files list
print(audio_files)
#res = [f for f in audio_files if "20240702" in f] 
#res = [f for f in audio_files if "20240725" in f] 
#res2 =  [f for f in res if  "MD_26Jul2024"  in f]

if __name__ == '__main__':
    scores = model.predict(audio_files, clip_overlap = 0.15, num_workers = 20, batch_size=36)
    scores.head()
    scores.to_csv("/Volumes/TundraBUZZ/outputs/recognizer_outputs/test_ARUQ10_2025/predict_score_ARUQ10_quick.csv") #set this to where you want to save the predicitons
