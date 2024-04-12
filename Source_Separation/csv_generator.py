from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import SignalDistortionRatio
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.batch import PaddedBatch
import pandas as pd
import numpy as np
import torch


class source_mix_dataset(Dataset):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source_1_path = self.df['source_1_path'].iloc[idx]
        #source1 = read_audio(source_1_path).squeeze()
        source_2_path = self.df['source_2_path'].iloc[idx]
        #source2 = read_audio(source_2_path).squeeze()
        mixture_path = self.df['mixture_path'].iloc[idx]
        #mixture = read_audio(mixture_path).squeeze()
        mixture_id = self.df['mixture_ID'].iloc[idx]
        length = self.df['length'].iloc[idx]
        return mixture_id, mixture_path, source_1_path, source_2_path, length



def create_csv(dataset, csv_file, idx_shift):
    df = pd.DataFrame(columns = ["ID", "duration", "mix_wav", "mix_wav_format", "mix_wav_opts", "s1_wav", "s1_wav_format", "s1_wav_format", "s2_wav", "s2_wav_format", "s2_wav_opts"])
    for idx, item in enumerate(dataset):
        row = [idx+idx_shift, 1.0, item[1], "wav", None, item[2], "wav", None, item[3], "wav", None]
        df.loc[idx] = row
    df.to_csv(csv_file, index=False)
    return

if __name__ == "__main__":
    file_path = '/home/ubuntu/sapa/Source_Separation/dataset/Libri2Mix/wav8k/min/metadata/mixture_test_mix_clean.csv'
    audio_dataset = source_mix_dataset(file_path)
    
    generator = torch.Generator().manual_seed(42)
    train_audio_set, test_audio_set = random_split(audio_dataset, [0.7, 0.3], generator=generator)
    print("Creating CSV file for train and test splits for min-mixture_test_mix_clean in wav8......")
    create_csv(train_audio_set, "train_min_mixture_test_mix_clean_wav8.csv", 9000)
    create_csv(test_audio_set, "test_min_mixture_test_mix_clean_wav8.csv", 3000)


    file_path = '/home/ubuntu/sapa/Source_Separation/dataset/Libri2Mix/wav8k/max/metadata/mixture_test_mix_clean.csv'
    audio_dataset = source_mix_dataset(file_path)
    
    generator = torch.Generator().manual_seed(42)
    train_audio_set, test_audio_set = random_split(audio_dataset, [0.7, 0.3], generator=generator)
    print("Creating CSV file for train and test splits for max-mixture_test_mix_clean in wav8......")
    create_csv(train_audio_set, "train_max_mixture_test_mix_clean_wav8.csv", 13000)
    create_csv(test_audio_set, "test_max_mixture_test_mix_clean_wav8.csv", 7000)

    df_min_train = pd.read_csv("train_min_mixture_test_mix_clean_wav8.csv")
    df_max_train = pd.read_csv("train_max_mixture_test_mix_clean_wav8.csv")
    df_train = pd.concat([df_min_train, df_max_train], axis=0)
    print("Creating CSV file for combined train for both min and max in wav8......")
    df_train.to_csv("train_min_max.csv", index=False)

