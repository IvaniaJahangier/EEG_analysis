#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# @Author: Ivania Jahangier
# @Date: 12-10-2021 
# ---------------------------------------------------------------------------

#loading packages
import pandas as pd
import seaborn as sns
import glob
import os as os
import sklearn
from picard import picard
from turtle import title
import numpy
import numpy as np
from scipy.special import logsumexp

import pathlib
from pathlib import Path

import PyQt5
import matplotlib.pyplot
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

import mne
from mne.time_frequency import tfr_morlet, tfr_array_morlet
from mne import event
from mne.io import concatenate_raws, read_raw_edf
from mne import concatenate_events, find_events
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.baseline import rescale 
from mne.stats import bootstrap_confidence_interval
from mne.viz import plot_topomap #newest
from mne.event import define_target_events


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from numpy import unravel_index
from statsmodels.stats.anova import AnovaRM
import pingouin as pg 
import scipy.stats as stats
import statsmodels.stats.multitest as smm

#allows an entire array to be visisble in jupyter notebook 
np.set_printoptions( threshold= 50) #np. inf) 
pd.options.display.max_rows = 100


# In[ ]:


# Global variables 
reject_criteria = dict(eeg=150e-6)  # 100uV
flat_criteria = dict(eeg=1e-7)  # 1uV
interval = (-0.5, 0)
sfreq = 1000
tmin = 0 
tmax = 3
fill_na = 99
wave_cycles = 5
frequenciesa1 = np.arange(8, 11, 1) # a1 band
frequenciesa2 = np.arange(11, 14, 1) # a2 band
frequenciesa = np.arange(4, 19, 1) # for the plot
picks = ['PO7', 'PO8']
number_of_trials = 240 #for every RVF/LVF per condition (240x4=960 trials)
shape = (0,2501) 
dtype = float




# In[ ]:


#creating lists that will be used during the analysis of both conditions
PRO_files = [] 
RETRO_files = [] 
corr_list = []
corresponding = []
subjectnumber = []
condition = []
pc_list = []
p_correspondence = []  

# These lists will be used for the pre-cue condition. 
# Create the empty arrays to store the data later on
a1_PRO_L_PO7 = np.empty(shape = shape, dtype = dtype)
a1_PRO_L_PO8 = np.empty(shape = shape, dtype = dtype)
a1_PRO_R_PO7 = np.empty(shape = shape, dtype = dtype)
a1_PRO_R_PO8 = np.empty(shape = shape, dtype = dtype)
a2_PRO_L_PO7 = np.empty(shape = shape, dtype = dtype)
a2_PRO_L_PO8 = np.empty(shape = shape, dtype = dtype)
a2_PRO_R_PO7 = np.empty(shape = shape, dtype = dtype)
a2_PRO_R_PO8 = np.empty(shape = shape, dtype = dtype)

# These lists will be used for the post-cue condition. 
# Create the empty arrays to store the data later on
a1_RETRO_L_PO7 = np.empty(shape = shape, dtype = dtype)
a1_RETRO_L_PO8 = np.empty(shape = shape, dtype = dtype)
a1_RETRO_R_PO7 = np.empty(shape = shape, dtype = dtype)
a1_RETRO_R_PO8 = np.empty(shape = shape, dtype = dtype)
a2_RETRO_L_PO7 = np.empty(shape = shape, dtype = dtype)
a2_RETRO_L_PO8 = np.empty(shape = shape, dtype = dtype)
a2_RETRO_R_PO7 = np.empty(shape = shape, dtype = dtype)
a2_RETRO_R_PO8 = np.empty(shape = shape, dtype = dtype)


# # LPS functions

# In[ ]:


#Here we are creating a function that calculates the LPS indices
#the formula is adapted from Van der Lubbe and Utzerath (2013)

def Left_cues (L_ipsi, L_contra):
    LPS_Left_cues = (L_ipsi-L_contra)/(L_ipsi+L_contra)                                 
    return LPS_Left_cues
    
def Right_cues (R_ipsi, R_contra):
    LPS_Right_cues = (R_ipsi-R_contra)/(R_ipsi+R_contra) 
    return LPS_Right_cues
                                       
                                       
def N_LPS(L_ipsi, L_contra, R_ipsi, R_contra):
    LPS_Left_cues = (L_ipsi-L_contra)/(L_ipsi+L_contra)
    LPS_Right_cues = (R_ipsi-R_contra)/(R_ipsi+R_contra)
    final_LPS = (LPS_Left_cues+LPS_Right_cues)/2
    return final_LPS


# # ICA analysis

# In[ ]:


# this way the ICA doesn't get rerun everytime the code is run
completeICA = False 

#Change the working directory to open the raw data
working_directory = os.chdir("C:/Users/ivani/Desktop/dir_eeg_analysis/EEG.data")

#Create an empty list to store the raw files 
ICA_solutions = [] # move to the lists section
for item in glob.glob("*.eeg"):
    ICA_solutions.append(item)
    
for item in ICA_solutions:
    if completeICA:
        raw = mne.io.read_raw_brainvision(item, preload=True)
        anno = mne.read_annotations(item, sfreq='auto', uint16_codec=None)
    
#plotting the data to check for bad channels
        raw_RETRO.plot()
        raw_RETRO.plot_psd(fmax=500)
    
    
# Setting the montage to the extended 10-20system, which is how we indicate how the electrodes were positioned 
# On the head. The data was recorded by setting the channel TP8 as the reference electrode. 

        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        raw = mne.add_reference_channels(raw_PRO, ref_channels=['TP8'])
        raw_PRO.set_eeg_reference(ref_channels='average')
    
# Plot to show the waves and their source on the head
        raw_PRO.plot_sensors(kind='topomap', show_names=True, title='ProPlot')

# We are filtering all major frequencies to enhance the quality of the data. 
# Since these frequency drifts can make it hard to create an ICA solution. 
        raw.load_data().filter(l_freq=0.1, h_freq=30)
        print('step 1 preparation complete')
    
# Starting the ICA
# Since the ICA can change our raw data We start of with creating a copy of our data
        raw_copy= raw.copy()
        raw_copy.load_data().filter(l_freq=0.1, h_freq=30)
    
# Variables that are used in the ICA
        nu_components = 63
        algorithm = 'fastica'
        random_seed = 91 #Setting a random state ensures that we get the same random value for every train and test datatesets. Otherwise it would set to non everytime and generate different values each time.
        ica = mne.preprocessing.ICA( n_components= nu_components, method= algorithm, random_state= random_seed )
        ica.fit(raw_copy)
        
# Instead of manually selecting which ICs to exclude, we use dedicated EOG sensors as a "pattern" to check the ICs against
        ica.exclude =[] #first we make an empty exclude list
        eog_indices, eog_scores = ica.find_bads_eog(raw_PRO_copy, ['hEOG', 'vEOG'])#automatically find the ICs that best match the EOG signal 
        ica.exclude = eog_indices#excludes artefacts matching eog signals that are added to the exclude list

# Barpolt of ICA component "EOG" match scores
        fig_1 = ica.plot_scores(eog_scores)
        plt.close(fig_1)
        fig_1.savefig(r'C:/Users/ivani/Desktop/Resultseeg/ICA_PRO_component_Score.png', overwrite=False)

# Plot diagnostics
        fig_2 = ica.plot_properties(raw_PRO_copy, picks=eog_indices) #save this image

# Visual presentation ICA components on head
        fig_4 = ica.plot_components(ch_type = 'eeg')


# Plot ICs applied to raw data, with EOG matches highlighted 
        fig_3 = ica.plot_sources(raw_PRO_copy)
        fig_3.figure.savefig(r'C:/Users/ivani/Desktop/Resultseeg/ICA_PRO_plot_sources.png', overwrite=False)
        
# Apply the ICA
        ica.apply(raw_copy)
    
# Final check of raw data, here the data should be full cleaned
        fig_5 = raw_PRO_copy.plot(title='Final check')
        fig_5.figure.savefig(r'C:/Users/ivani/Desktop/Resultseeg/raw_PRO_final_check.png', overwrite=True)
        
# If the data is fully cleaned, the ICA solution can be saved
        raw_copy.save(r'C:/Users/ivani/Desktop/Resultseeg/raw_cleaned.fif', overwrite=True) 
    else: 
        print('Ica is completed')
    



    

    


# # Pre-cue analysis 

# In[ ]:


# This loop consist of two main parts
# he first part involved creating epochs, TFRs and saving the data to arrays 
# This conderns the PO7/PO8 electrodes for the a1 and a2 band
# This data will be used to calculate the LPS later on 
# The second part involves the data analysis of the behavioral data. 
# The events are merged and saved to lists that will be used in a pandas dataframe

#Changing the working directory to find the cleaned files 
working_directory = os.chdir("C:/Users/ivani/Desktop/dir_eeg_analysis/ICA_Solutions/PRO")

# Saving the data for this condition in one list.
for item in glob.glob("*.fif"):
    PRO_files.append(item)



# Variables needed for this section
subject = 1
current_condition = 'pre-cue'


for item in PRO_files:
    PRO_raw = mne.io.read_raw_fif(item, preload=True)
    
    events_PRO, event_dict_pro = mne.events_from_annotations(PRO_raw)
    
    events_PRO, notNeeded = mne.events_from_annotations(PRO_raw)   

    event_dict_PRO = {"fixation start": 2, 'cue': 3,
                  'target': 4, 'correctresponse/1': 101,
                  'correctresponse/2': 102, 'correctresponse/3': 103,
                  'correctresponse/4': 104, 'incorrectresponse': 105,
                  'PRO_LVF/L': 11, 'PRO_LVF/R': 12,
                  'PRO_RVF/L': 21, 'PRO_RVF/R': 22}


#creating epochs
    epochs_PRO = mne.Epochs(PRO_raw, events_PRO, event_id=event_dict_PRO, tmin=-0.5, tmax=2.0, preload=True) 
    epochs_PRO.drop_bad(reject=reject_criteria, flat=flat_criteria)
    epochs_PRO.apply_baseline(interval)
    
# The TFR analysis is performed twice, one for the a1 and one for the a2 band. 

# We start with the manipulationg in the a1 band 
# Using a morlet wavelet
    L_tfr_a1 = tfr_morlet(epochs_PRO['PRO_LVF'], freqs=frequenciesa1, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')  
    
    R_tfr_a1 = tfr_morlet(epochs_PRO['PRO_RVF'], freqs=frequenciesa1, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')

    
# Save the data from the TFR into separate arrays 
    a1_PO7 = L_tfr_a1.data[0,:,:]
    a1_PO8 = L_tfr_a1.data[1,:,:]
    a1_PO72 = R_tfr_a1.data[0,:,:]
    a1_PO82 = R_tfr_a1.data[1,:,:]
    

# Takes the average from the 4 frequencies for the band per time point (2501 points) and computes the mean. 
# Creating one mean for the entire frequency band 
    m_a1_PO7 = np.mean(a1_PO7, axis=0, keepdims=True) 
    m_a1_PO8 = np.mean(a1_PO8, axis=0, keepdims=True)
    m_a1_PO72 = np.mean(a1_PO72, axis=0, keepdims=True)
    m_a1_PO82 = np.mean(a1_PO82, axis=0, keepdims=True)
        
    
#saving the mean values to arrays
    a1_PRO_L_PO7 = np.append(a1_PRO_L_PO7, m_a1_PO7, axis=0)
    a1_PRO_L_PO8 = np.append(a1_PRO_L_PO8, m_a1_PO8, axis=0)
    a1_PRO_R_PO7 = np.append(a1_PRO_R_PO7, m_a1_PO72, axis=0)
    a1_PRO_R_PO8 = np.append(a1_PRO_R_PO8, m_a1_PO82, axis=0)

# We start with the manipulationg in the a2 band 
# Using a morlet wavelet
    L_tfr_a2 = tfr_morlet(epochs_PRO['PRO_LVF'], freqs=frequenciesa2, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')  # power ipv complex
    
    R_tfr_a2= tfr_morlet(epochs_PRO['PRO_RVF'], freqs=frequenciesa2, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')
    
# Save the data from the TFR into separate arrays 
    a2_PO7 = L_tfr_a2.data[0,:,:]
    a2_PO8 = L_tfr_a2.data[1,:,:]
    a2_PO72 = R_tfr_a2.data[0,:,:]
    a2_PO82 = R_tfr_a2.data[1,:,:]
    
# Takes the average from the 4 frequencies for the band per time point (2501 points) and computes the mean. 
# Creating one mean for the entire frequency band 
    m_a2_PO7 = np.mean(a2_PO7, axis=0, keepdims=True) 
    m_a2_PO8 = np.mean(a2_PO8, axis=0, keepdims=True)
    m_a2_PO72 = np.mean(a2_PO72, axis=0, keepdims=True)
    m_a2_PO82 = np.mean(a2_PO82, axis=0, keepdims=True)
        
    
# Saving the mean values to arrays
    a2_PRO_L_PO7 = np.append(a2_PRO_L_PO7, m_a2_PO7, axis=0)
    a2_PRO_L_PO8 = np.append(a2_PRO_L_PO8, m_a2_PO8, axis=0)
    a2_PRO_R_PO7 = np.append(a2_PRO_R_PO7, m_a2_PO72, axis=0)
    a2_PRO_R_PO8 = np.append(a2_PRO_R_PO8, m_a2_PO82, axis=0)


# Now we start with creating the data for the pandas dataframe 
# This will be done for both the corresponding and non-corresponding trials
# First the correct answers are all merged 
# Then the corresponding trials (visual fieldxhandedness) are merged
# Lastly the non-corresponding trials are merged

    events_PRO = mne.merge_events(events_PRO, [101,102,103,104], 80, replace_events=False) 
    events_PRO = mne.merge_events(events_PRO, [11,22], 13, replace_events=False) 
    events_PRO = mne.merge_events(events_PRO, [12,21], 14, replace_events=False)
    
    
# These events sets up the reaction times for the corresponding correct responses
# Setting up the RTs. Which starts with the define-target function. 
# This allows us to define new targets based on co-occuring events and the time (in ms) between 
# the target and the reference id events

    c1_events, nan_rt1 = define_target_events(events = events_PRO, reference_id = 13, 
                                          target_id = 80, sfreq = sfreq, 
                                          tmin = tmin, tmax=3, new_id = 25, fill_na=fill_na)
    
# Now we remove the nan values from the list
    rt1 = nan_rt1[np.logical_not(np.isnan(nan_rt1))]    
    
# Calculate the percentage of correct answers 
    pc = len(rt1)/number_of_trials*100


# Appending the RT, corresponding side, participantnumber, condition,and pc 
    corr_list.append(np.mean(rt1, axis=0)) 
    corresponding.append('corr') 
    subjectnumber.append(subject)
    condition.append(current_condition)
    pc_list.append(pc)
    
# These events sets up the reaction times for the corresponding correct responses
# Setting up the RTs. Which starts with the define-target function. 
# This allows us to define new targets based on co-occuring events and the time (in ms) between 
# the target and the reference id events
    c2_events, nan_rt2 = define_target_events(events = events_PRO, reference_id = 14, 
                                          target_id = 80, sfreq = sfreq, 
                                          tmin = tmin, tmax=3, new_id = 15, fill_na=fill_na)
# Now we remove the nan values from the list
    rt2 = nan_rt2[np.logical_not(np.isnan(nan_rt2))]   

# Calculate the percentage of correct answers
    pc = len(rt2)/number_of_trials*100
    
# Appending the RT, non-corresponding side, participantnumber, condition,and pc 
    corr_list.append(np.mean(rt2, axis=0)) 
    corresponding.append('noncorr')
    subjectnumber.append(subject)
    condition.append(current_condition)
    pc_list.append(pc)

    
    subject = subject+1


# ## post-cue analysis

# In[ ]:


# This loop consist of two main parts
# he first part involved creating epochs, TFRs and saving the data to arrays 
# This conderns the PO7/PO8 electrodes for the a1 and a2 band
# This data will be used to calculate the LPS later on 
# The second part involves the data analysis of the behavioral data. 
# The events are merged and saved to lists that will be used in a pandas dataframe

#Changing the working directory to find the cleaned files 
working_directory = os.chdir("C:/Users/ivani/Desktop/dir_eeg_analysis/ICA_Solutions/RETRO")


# Saving the data for this condition in one list.
for item in glob.glob("*.fif"):
    RETRO_files.append(item)


# Variables needed for this section
subject = 1
current_condition = 'post-cue'


for item in RETRO_files:
    RETRO_raw = mne.io.read_raw_fif(item, preload=True)
    
    event_dict_RETRO = {'fixation start': 2, 'Cue': 3,
                'target': 4, 'correctresponse/1': 101,
                'correctresponse/2': 102, 'correctresponse/3': 103,
                'correctresponse/4': 104, 'incorrectresponse': 105,
                'RETRO_LVF/L': 111, 'RETRO_LVF/R': 112,
                'RETRO_RVF/L': 121, 'RETRO_RVF/R': 122}

    
    events_RETRO, event_dict_retro = mne.events_from_annotations(RETRO_raw)
    
    events_RETRO, notNeeded = mne.events_from_annotations(RETRO_raw)   
    print(events_RETRO)
    
#creating epochs
    epochs_RETRO = mne.Epochs(RETRO_raw, events_RETRO, event_id=event_dict_RETRO, tmin=-0.5, tmax=2.0) 
    epochs_RETRO.drop_bad(reject=reject_criteria, flat=flat_criteria)
    epochs_RETRO.apply_baseline(interval)

# The TFR analysis is performed twice, one for the a1 and one for the a2 band. 
# We start with the manipulationg in the a1 band 
# Using a morlet wavelet
    L_tfr_a1 = tfr_morlet(epochs_RETRO['RETRO_LVF'], freqs=frequenciesa1, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power') 
    
    R_tfr_a1= tfr_morlet(epochs_RETRO['RETRO_RVF'], freqs=frequenciesa1, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')
    

# Save the data from the TFR into separate arrays 
    a1_PO7 = L_tfr_a1.data[0,:,:]
    a1_PO8 = L_tfr_a1.data[1,:,:]
    a1_PO72 = R_tfr_a1.data[0,:,:]
    a1_PO82 = R_tfr_a1.data[1,:,:]
    

# Takes the average from the 4 frequencies for the band per time point (2501 points) and computes the mean. 
# Creating one mean for the entire frequency band 
    m_a1_PO7 = np.mean(a1_PO7, axis=0, keepdims=True) 
    m_a1_PO8 = np.mean(a1_PO8, axis=0, keepdims=True)
    m_a1_PO72 = np.mean(a1_PO72, axis=0, keepdims=True)
    m_a1_PO82 = np.mean(a1_PO82, axis=0, keepdims=True)
        
    
#saving the mean values to arrays
    a1_RETRO_L_PO7 = np.append(a1_RETRO_L_PO7, m_a1_PO7, axis=0)
    a1_RETRO_L_PO8 = np.append(a1_RETRO_L_PO8, m_a1_PO8, axis=0)
    a1_RETRO_R_PO7 = np.append(a1_RETRO_R_PO7, m_a1_PO72, axis=0)
    a1_RETRO_R_PO8 = np.append(a1_RETRO_R_PO8, m_a1_PO82, axis=0)
    
# We start with manipulationg the a2 band 
# Using a morlet wavelet
    L_tfr_a2 = tfr_morlet(epochs_RETRO['RETRO_LVF'], freqs=frequenciesa2, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power') 
    
    R_tfr_a2= tfr_morlet(epochs_RETRO['RETRO_RVF'], freqs=frequenciesa2, picks=picks, return_itc=False,
                          n_cycles=wave_cycles, average = True, output= 'power')
    
# Save the data from the TFR into separate arrays 
    a2_PO7 = L_tfr_a2.data[0,:,:]
    a2_PO8 = L_tfr_a2.data[1,:,:]
    a2_PO72 = R_tfr_a2.data[0,:,:]
    a2_PO82 = R_tfr_a2.data[1,:,:]
    

# Takes the average from the 4 frequencies for the band per time point (2501 points) and computes the mean. 
# Creating one mean for the entire frequency band 
    m_a2_PO7 = np.mean(a2_PO7, axis=0, keepdims=True) 
    m_a2_PO8 = np.mean(a2_PO8, axis=0, keepdims=True)
    m_a2_PO72 = np.mean(a2_PO72, axis=0, keepdims=True)
    m_a2_PO82 = np.mean(a2_PO82, axis=0, keepdims=True)
        
    
# Saving the mean values to arrays
    a2_RETRO_L_PO7 = np.append(a2_RETRO_L_PO7, m_a2_PO7, axis=0)
    a2_RETRO_L_PO8 = np.append(a2_RETRO_L_PO8, m_a2_PO8, axis=0)
    a2_RETRO_R_PO7 = np.append(a2_RETRO_R_PO7, m_a2_PO72, axis=0)
    a2_RETRO_R_PO8 = np.append(a2_RETRO_R_PO8, m_a2_PO82, axis=0)



# Now we start with creating the data for the pandas dataframe 
# This will be done for both the corresponding and non-corresponding trials
# First the correct answers are all merged 
# Then the corresponding trials (visual fieldxhandedness) are merged
# Lastly the non-corresponding trials are merged

    events_RETRO = mne.merge_events(events_RETRO, [101,102,103,104], 80, replace_events=False)
    events_RETRO = mne.merge_events(events_RETRO, [111,122], 113, replace_events=False)
    events_RETRO = mne.merge_events(events_RETRO, [112,121], 114, replace_events=False)
 

# These events set up the reaction times for the corresponding correct responses 
# Setting up the RTs. Which starts with the define-target function. 
# This allows us to define new targets based on co-occuring events and the time (in ms) between 
# the target and the reference id events
    c1_events, nan_rt1 = define_target_events(events = events_RETRO, reference_id = 113, 
                                          target_id = 80, sfreq = sfreq, 
                                          tmin = tmin, tmax=tmax, new_id = 25, fill_na=fill_na)
#now we remove the nan values from the list
    rt1 = nan_rt1[np.logical_not(np.isnan(nan_rt1))]

# Calculate the percentage of correct answers 
    pc = len(rt1)/number_of_trials*100
    
# Appending the RT, corresponding side, participantnumber, condition,and pc 
    corr_list.append(np.nanmean(rt1, axis=0))
    corresponding.append('corr') 
    subjectnumber.append(subject)
    condition.append(current_condition)
    pc_list.append(pc)
    
# These events sets up the reaction times for the corresponding correct responses
# Setting up the RTs. Which starts with the define-target function. 
# This allows us to define new targets based on co-occuring events and the time (in ms) between 
# the target and the reference id events
    c2_events, nan_rt2 = define_target_events(events = events_RETRO, reference_id = 114, 
                                          target_id = 80, sfreq = sfreq, 
                                          tmin = tmin, tmax=tmax, new_id = 15, fill_na=fill_na)
   
# Now we remove the nan values from the list
    rt2 = nan_rt2[np.logical_not(np.isnan(nan_rt2))]
    
# Calculate the percentage of correct answers
    pc = len(rt2)/number_of_trials*100
    
# Appending the RT, non-corresponding side, participantnumber, condition,and pc 
    corr_list.append(np.nanmean(rt2, axis=0))
    corresponding.append('noncorr')
    subjectnumber.append(subject)
    condition.append(current_condition)
    pc_list.append(pc)
        
    subject = subject+1
    


# ## LPS construction

# In[ ]:


# construct the LPS for both conditions

#pre-cue a1
PRO_left_cues_a1 = Left_cues(a1_PRO_L_PO7,a1_PRO_L_PO8)
PRO_right_cues_a1 = Right_cues(a1_PRO_R_PO8, a1_PRO_R_PO7)
PRO_epoch_array_a1 = N_LPS(a1_PRO_L_PO7,a1_PRO_L_PO8,a1_PRO_R_PO8, a1_PRO_R_PO7)

#pre-cue a2
PRO_left_cues_a2 = Left_cues(a2_PRO_L_PO7,a2_PRO_L_PO8)
PRO_right_cues_a2 = Right_cues(a2_PRO_R_PO8, a2_PRO_R_PO7)
PRO_epoch_array_a2 = N_LPS(a2_PRO_L_PO7,a2_PRO_L_PO8,a2_PRO_R_PO8, a2_PRO_R_PO7)

#post-cue a1
RETRO_left_cues_a1 = Left_cues(a1_RETRO_L_PO7,a1_RETRO_L_PO8)
RETRO_right_cues_a1 = Right_cues(a1_RETRO_R_PO8, a1_RETRO_R_PO7,)
RETRO_epoch_array_a1 = N_LPS(a1_RETRO_L_PO7,a1_RETRO_L_PO8,a1_RETRO_R_PO8, a1_RETRO_R_PO7)

#post-cue a2
RETRO_left_cues_a2 = Left_cues(a2_RETRO_L_PO7,a2_RETRO_L_PO8)
RETRO_right_cues_a2 = Right_cues(a2_RETRO_R_PO8, a2_RETRO_R_PO7,)
RETRO_epoch_array_a2 = N_LPS(a2_RETRO_L_PO7,a2_RETRO_L_PO8,a2_RETRO_R_PO8, a2_RETRO_R_PO7)




# # LPS power plots 

# ## pre-cue  plots 

# In[ ]:


#creating the frame to plot both alpha bands in one plot 
#Because the values are mapped from the array they are plotted from [0,0] 
#we create some empty layers to allow the plot to be scaled more accurately 

PRO_epoch_array_a1_np = np.mean(PRO_epoch_array_a1, axis=0, keepdims=True)
PRO_epoch_array_a2_np = np.mean(PRO_epoch_array_a2, axis=0, keepdims=True)
PRO_epoch_array = np.vstack([PRO_epoch_array_a1_np, PRO_epoch_array_a2_np])

empty = np.zeros(shape = (2,2501), dtype = float)
PRO_epoch_array2 = np.vstack ([empty,PRO_epoch_array ])
PRO_epoch_array3 = np.vstack ([PRO_epoch_array2, empty ])


# In[ ]:


times =epochs_PRO.times
plot_max = np.max(np.max(PRO_epoch_array3))
plot_min = -plot_max

fig, ax = plt.subplots(1)
im = plt.imshow(PRO_epoch_array3,
           extent=[times[0], times[-1], frequenciesa[0], frequenciesa[-1]],
           aspect='auto', origin='lower', cmap='coolwarm', vmin=plot_min, vmax=plot_max)

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('LPS Pre-cue')
cb = fig.colorbar(im)
cb.set_label('Power')
plt.axvline(x=0, color = 'k', linewidth = 1)
plt.show()


# ## post-cue plot

# In[ ]:


#creating the frame to plot both alpha bands in one plot 
#Because the values are mapped from the array they are plotted from [0,0] 
#we create some empty layers to allow the plot to be scaled more accurately 


RETRO_epoch_array_a1_np = np.mean(RETRO_epoch_array_a1, axis=0, keepdims=True)
RETRO_epoch_array_a2_np = np.mean(RETRO_epoch_array_a2, axis=0, keepdims=True)

RETRO_epoch_array = np.vstack([RETRO_epoch_array_a1_np, RETRO_epoch_array_a2_np])
# creating an empty list 
empty = np.zeros(shape = (2,2501), dtype = float)
RETRO_epoch_array2 = np.vstack ([empty,RETRO_epoch_array ])
RETRO_epoch_array3 = np.vstack ([RETRO_epoch_array2, empty ])


# In[ ]:


times =epochs_RETRO.times
plot_max = np.max(np.max(RETRO_epoch_array3))
plot_min = -plot_max

fig, ax = plt.subplots(1)
im = plt.imshow(RETRO_epoch_array3,
           extent=[times[0], times[-1], frequenciesa[0], frequenciesa[-1]],
           aspect='auto', origin='lower', cmap='coolwarm', vmin=plot_min, vmax=plot_max)

plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.title('LPS post-cue')
cb = fig.colorbar(im)
cb.set_label('Power')

plt.axvline(x=0, color = 'k', linewidth = 1)
plt.show()


# # statistical analysis

# In[ ]:


# The statistical analysis consists of two main parts
# First we perform the LPS analysis to look for relevant time windows
# Then we perform a 2x2 repeated measures ANOVA 


# ## LPS analysis

# ###  pre-cue a1

# In[ ]:


# Creating 40 ms time intervals for the analysis
# Performing one sample one tailed t-tests to get an overview of the raw sig time windows
chunk_size = 40
sig_time_windows = [] 
for i in range(0, 2501, chunk_size):
    chunk = PRO_epoch_array_a1[:, i:i+chunk_size]
    chunk = chunk.flatten()
    tstat, pval = stats.ttest_1samp(chunk, popmean=0, alternative = 'greater')
    if pval1 <= 0.05:
        print('time window', + i,'till', + i+chunk_size, 'ms ', 'pval=',+pval)
    sig_time_windows.append(pval)


# In[ ]:


#calculating adjusted p-values 
chunk_size = 0
reject, pvals, alphacs, alphacb = smm.multipletests(list3, alpha=0.05, method='b', is_sorted=False, 
                                                    returnsorted=False)
for i in pvals:
    if i <= 0.05:
        print('time window', + chunk_size,'till', + chunk_size+40, 'ms ', 'pval=',+i )
    chunk_size = chunk_size+40


# In[ ]:


#calculate pvalue of the sig time window 
tstat, pval = stats.ttest_1samp(PRO_epoch_array_a1[:, 640:1480].flatten(), popmean=0, 
                                alternative = 'greater')
print('time window 140-980 ms', 'pval=',+pval, 'tstat=',+tstat )


# In[ ]:


#calculate the polarity of the sig time windows
pol_1 = np.mean(PRO_epoch_array_a1[:, 640:1480].flatten())
print('Polarity time window 140-980 ms = '+ str(pol_1))


# ### pre-cue a2

# In[ ]:


# Creating 40 ms time intervals for the analysis
# Performing one sample one tailed t-tests to get an overview of the raw sig time windows
chunk_size = 40
sig_time_windows = [] 
for i in range(0, 2501, chunk_size):
    chunk = PRO_epoch_array_a2[:, i:i+chunk_size]
    chunk = chunk.flatten()
    tstat, pval = stats.ttest_1samp(chunk, popmean=0, alternative = 'greater')
    if pval1 <= 0.05:
        print('time window', + i,'till', + i+chunk_size, 'ms ', 'pval=',+pval)
    sig_time_windows.append(pval)


# In[ ]:


#calculating adjusted p-values 
chunk_size = 0
reject, pvals, alphacs, alphacb = smm.multipletests(list3, alpha=0.05, method='b', is_sorted=False, 
                                                    returnsorted=False)
for i in pvals:
    if i <= 0.05:
        print('time window', + chunk_size,'till', + chunk_size+40, 'ms ', 'pval=',+i )
    chunk_size = chunk_size+40


# In[ ]:


#calculate pvalue of the sig time window 
tstat, pval = stats.ttest_1samp(PRO_epoch_array_a2[:, 640:1320].flatten(), popmean=0, 
                                alternative = 'greater')
print('time window 140-820 ms', 'pval=',+pval, 'tstat=',+tstat )


# In[ ]:


#calculate the polarity of the sig time windows
pol_1 = np.mean(PRO_epoch_array_a2[:, 640:1320].flatten())
print('Polarity time window 140-820 ms = '+ str(pol_1))


# ### post-cue a1

# In[ ]:


# Creating 40 ms time intervals for the analysis
# Performing one sample one tailed t-tests to get an overview of the raw sig time windows
chunk_size = 40
sig_time_windows = [] 
for i in range(0, 2501, chunk_size):
    chunk = RETRO_epoch_array_a1[:, i:i+chunk_size]
    chunk = chunk.flatten()
    tstat, pval = stats.ttest_1samp(chunk, popmean=0, alternative = 'greater')
    if pval1 <= 0.05:
        print('time window', + i,'till', + i+chunk_size, 'ms ', 'pval=',+pval)
    sig_time_windows.append(pval)


# In[ ]:


#calculating adjusted p-values 
chunk_size = 0
reject, pvals, alphacs, alphacb = smm.multipletests(list3, alpha=0.05, method='b', is_sorted=False, 
                                                    returnsorted=False)
for i in pvals:
    if i <= 0.05:
        print('time window', + chunk_size,'till', + chunk_size+40, 'ms ', 'pval=',+i )
    chunk_size = chunk_size+40


# In[ ]:


#calculate pvalue of the sig time window 
tstat, pval = stats.ttest_1samp(RETRO_epoch_array_a1[:, 600:1480].flatten(), popmean=0,
                                alternative = 'greater')
print('time window 100-980 ms', 'pval=',+pval, 'tstat=',+tstat )


# In[ ]:


#calculate the polarity of the sig time windows
pol_1 = np.mean(RETRO_epoch_array_a2[:, 840:1480].flatten())
print('Polarity time window 100-980 ms = '+ str(pol_1))


# ### post-cue a2

# In[ ]:


#variables
chunk_size = 40
sig_time_windows = [] 

# Create 40 ms time intervals for the analysis
# Perform one sample one tailed t-tests to get an overview of the raw sig time windows
# Save sig time windows in one list 
for i in range(0, 2501, chunk_size):
    chunk = RETRO_epoch_array_a2[:, i:i+chunk_size]
    chunk = chunk.flatten()
    tstat, pval = stats.ttest_1samp(chunk, popmean=0, alternative = 'greater')
    if pval1 <= 0.05:
        print('time window', + i,'till', + i+chunk_size, 'ms ', 'pval=',+pval)
    sig_time_windows.append(pval)


# In[ ]:


#calculating adjusted p-values 
chunk_size = 0
reject, pvals, alphacs, alphacb  = smm.multipletests(list3, alpha=0.05, method='b', is_sorted=False, 
                                                     returnsorted=False)
for i in pvals:
    if i <= 0.05:
        print('time window', + chunk_size,'till', + chunk_size+40, 'ms ', 'pval=',+i )
    chunk_size = chunk_size+40


# In[ ]:


#calculate pvalue of the sig time window 
tstat, pval = stats.ttest_1samp(RETRO_epoch_array_a2[:, 640:760].flatten(), popmean=0, 
                                alternative = 'greater')
print('time window 140-260 ms', 'pval=',+pval, 'tstat=',+tstat )

#calculate pvalue of the sig time window 
tstat, pval = stats.ttest_1samp(RETRO_epoch_array_a2[:, 840:1480].flatten(), popmean=0, 
                                alternative = 'greater')
print('time window 340-980 ms', 'pval=',+pval, 'tstat=',+tstat )


# In[ ]:


#calculate the polarity of the sig time windows
pol_1 = np.mean(RETRO_epoch_array_a2[:, 640:760].flatten())
print('Polarity time window 140-760 ms = '+ str(pol_1))

pol_2 = np.mean(RETRO_epoch_array_a2[:, 840:1480].flatten())
print('Polarity time window 340-980 ms = '+ str(pol_2))


# ## ANOVA 

# In[ ]:


#Create the pandas dataframe 
df = pd.DataFrame({'id': subjectnumber, 
                  'RT': corr_list, 
                  'condition': condition,
                  'corr_side': corresponding,
                  'pc':pc_list})
df


# In[ ]:


#test the sphericity of the data 
pg.sphericity(df, dv='pc', subject='id', within=['condition','corr_side'])
pg.sphericity(df, dv='RT', subject='id', within=['condition','corr_side'])


# In[ ]:


#calculate the means of the RT and PC
df.groupby(['condition', 'corr_side'])['RT'].mean()
df.groupby(['condition', 'corr_side'])['pc'].mean()


# In[ ]:


# Repeated measures ANOVA 2x2 factors for RT
pg.rm_anova(dv='RT', within=['condition', 'corr_side'], subject = 'id', data =df, detailed = True)


# In[ ]:


# Repeated measures ANOVA 2x2 factors for pc
pg.rm_anova(dv='pc', within=['condition', 'corr_side'], subject = 'id', data =df, detailed = True)


# In[ ]:


# PC pre-cue condition
# Filter the dataframe for the conditionxpc t-tests for correspondence
correspondence = df[df["condition"] =='pre-cue']
correspondence = correspondence['RT']

# One sample t test for simon effect(correspondence)
tstat, pval = stats.ttest_1samp(correspondence, popmean=0, alternative = 'greater')
p_correspondence.append(pval)


# In[ ]:


# PC pre-cue condition
# Filter the dataframe for the conditionxpc t-tests for correspondence
correspondence = df[df["condition"] =='post-cue']
correspondence = correspondence['RT']

# One sample t test for simon effect(correspondence)
tstat, pval = stats.ttest_1samp(correspondence, popmean=0, alternative = 'greater')
p_correspondence.append(pval)


# In[ ]:


# PC pre-cue condition
# Filter the dataframe for the conditionxpc t-tests for correspondence
correspondence = df[df["condition"] =='pre-cue']
correspondence = correspondence['pc']

# One sample t test for simon effect(correspondence)
tstat, pval = stats.ttest_1samp(correspondence, popmean=0, alternative = 'greater')
p_correspondence.append(pval)


# In[ ]:


# PC post-cue condition
# Filter the dataframe for the conditionxpc t-tests for correspondence
correspondence = df[df["condition"] =='post-cue']
correspondence = correspondence['pc']

# One sample t test for simon effect(correspondence)
tstat, pval = stats.ttest_1samp(correspondence, popmean=0, alternative = 'greater')
p_correspondence.append(pval)


# In[ ]:


# Correct the pvalues
reject, pvals, alphacs, alphacb  = smm.multipletests(p_correspondence, alpha=0.05, method='b', 
                                                     is_sorted=False, returnsorted=False)

