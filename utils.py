import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, AmplitudeToDB

import matplotlib.pyplot as plt

PATH = '../hms-original'
TARGET = 'expert_consensus'
target_col = ['gpd_vote', 'grda_vote', 'lpd_vote', 'lrda_vote', 'other_vote', 'seizure_vote']
signal_col = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
montage = ['LL','LP','RP','RR']
double_banana_montage = [['Fp1','F7','T3','T5','O1'],
                         ['Fp1','F3','C3','P3','O1'],
                         ['Fp2','F8','T4','T6','O2'],
                         ['Fp2','F4','C4','P4','O2']]

class EEGDataset(Dataset):
    def __init__(self, size=1, val_set=True, transpose=True, transform=False):
        self.train_df = pd.read_csv(f'{PATH}/train.csv')
        self.eeg_ids = self.train_df['eeg_id'].unique()
        self.val_set = val_set

        # split train and validation set
        self.train_egg_ids, self.val_egg_ids = self.split_train_val(self.eeg_ids)

        # split train set into smaller size for faster training
        self.t_length = len(self.train_egg_ids) if size == 1 else int(len(self.train_egg_ids) // (1 / size))
        if size != 1:
            self.train_egg_ids = np.random.choice(self.train_egg_ids, self.t_length, replace=False)
        
        self.transpose = transpose
        self.transform = Spectrogram(n_fft=512, hop_length=64) if transform else None
        
        self.le = LabelEncoder()
        self.train_df[TARGET] = self.le.fit_transform(self.train_df[TARGET])

    def __len__(self):
        return self.t_length if not self.val_set else len(self.val_egg_ids)

    def __getitem__(self, idx):
        if not self.val_set:
            eeg_ids = self.train_egg_ids
        else:
            eeg_ids = self.val_egg_ids

        id = eeg_ids[idx]
        random_sample = self.train_df[self.train_df['eeg_id'] == id].sample()
        offset = int(random_sample['eeg_label_offset_seconds'].item())
        class_prob = random_sample[target_col] / random_sample[target_col].sum(axis=1).item()

        df = self.read_data(id, offset)
        filtered_data = self.butter_lowpass_filter(df)
        cliped_data = np.clip(filtered_data.copy(), -1024, 1024) / 32
        pt_data = torch.Tensor(cliped_data)

        if self.transpose:
            pt_data = pt_data.T

        if self.transform:
            pt_data = self.transform(pt_data)
            pt_data = torch.nan_to_num(pt_data)

        return pt_data, random_sample[TARGET].item(), torch.Tensor(class_prob.values[0])
    
    def split_train_val(self, eeg_ids, val_size=0.2):
        np.random.shuffle(eeg_ids)
        split = int(len(eeg_ids) * (1 - val_size))
        train_ids, val_ids = eeg_ids[:split], eeg_ids[split:]
        return train_ids, val_ids

    def read_data(self, id, offset):
        df = pd.read_parquet(f'{PATH}/train_eegs/{id}.parquet')
        df = df.fillna(0)
        df = df.iloc[offset*200:(offset+50)*200]
        df = df[signal_col]

        # check for NaN values
        check_nan = df.isna().any(axis=1)
        if check_nan.sum() > 0:
            print(f'eeg id: {id}')
        
        return df
    
    def butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

class EEGCleanDataset(Dataset):
    def __init__(self, df, vote=1, size=1, transpose=True, transform=False):
        # filter out rows with less than n votes
        if isinstance(vote, int):
            df = df[df[target_col].sum(axis=1) >= vote]
        elif isinstance(vote, tuple):
            mask = (df[target_col].sum(axis=1) >= vote[0]) & (df[target_col].sum(axis=1) < vote[1])
            len = (df[target_col].sum(axis=1) >= vote[1]).sum()
            dif = mask.sum() - len
            df = df[mask]
            if dif > 0:
                df = df.sample(len)
            
        # split train set into smaller size for faster training
        if size == 1:
            self.df = df
        else:
            self.df = df[:int(len(df) * size)]
        
        self.transpose = transpose
        self.transform = Spectrogram(n_fft=512, hop_length=64) if transform else None
        
        self.le = LabelEncoder()
        self.df.loc[:, TARGET] = self.le.fit_transform(self.df[TARGET])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].astype('int64')
        id = row['eeg_id']
        offset = row['eeg_label_offset_seconds']
        class_prob = row[target_col] / row[target_col].sum()
        
        eeg_df = self.read_data(id, offset)
        filtered_data = self.butter_lowpass_filter(eeg_df)
        cliped_data = np.clip(filtered_data.copy(), -1024, 1024) / 32
        pt_data = torch.Tensor(cliped_data)

        if self.transpose:
            pt_data = pt_data.T

        if self.transform:
            pt_data = self.transform(pt_data)
            pt_data = torch.nan_to_num(pt_data)
        
        return pt_data, row[TARGET], torch.Tensor(class_prob.values)

    def read_data(self, id, offset):
        df = pd.read_parquet(f'{PATH}/train_eegs/{id}.parquet')
        df = df.fillna(0)
        df = df.iloc[offset*200:(offset+50)*200]
        df = df[signal_col]

        # check for NaN values
        check_nan = df.isna().any(axis=1)
        if check_nan.sum() > 0:
            print(f'eeg id: {id}')

        return df
    
    def butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

def plot_eegs(eeg, end_time=50, nrows=16, ncols=1):
    assert nrows * ncols == 16, 'nrows * ncols should be equal to 16'
    
    time = np.linspace(0, end_time, end_time*200)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 50))
    axes = axes.flatten()

    eeg = eeg.squeeze(0).numpy()
    for i in range(16):
        eeg_data = eeg[i, :end_time*200].T
        axes[i].plot(time, eeg_data)
        axes[i].set_title(f'{signal_col[i]}')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_specs(eeg):
    spectrogram_db = AmplitudeToDB()(eeg[0, 0])

    spec_data = spectrogram_db.numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(spec_data, cmap='viridis', aspect='auto', origin='lower')
    plt.title('Spectrogram of EEG Data')
    plt.xlabel('Time')
    plt.ylabel('Frequency Bin')
    plt.colorbar(label='dB')
    plt.show()

def rebuild_train_df(train_df, path_to_file):
    train_df['eeg_and_votes'] = train_df[['eeg_id'] + target_col].astype('str').agg('_'.join, axis=1)

    new = []
    for i in train_df['eeg_and_votes'].unique():
        rows = train_df[train_df['eeg_and_votes'] == i]
        offsets = rows['spectrogram_label_offset_seconds'].astype(int).values
        dist = np.zeros(offsets.max() + 600)
        for offset in offsets:
            dist[offset:offset + 600] += 1
        best_idx = np.argmax([dist[offset:offset+600].sum() for offset in offsets])
        new.append(rows.iloc[best_idx])
    
    # 17089 rows
    new = pd.concat(new, axis=1).T.drop_duplicates(subset='eeg_id').reset_index(drop=True)
    del new['eeg_and_votes']
    
    # save
    new.to_csv(path_to_file, index=False)