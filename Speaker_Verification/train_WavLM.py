from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import pipeline
from transformers import AutoProcessor, AutoModel, get_scheduler
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from speechbrain.dataio.dataio import read_audio
import librosa
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_curve

from transformers import AutoFeatureExtractor, UniSpeechSatForXVector, WavLMForXVector, Wav2Vec2ForXVector



class kanbath_dataset(Dataset):
    def __init__(self, file_path, transforms=None, flag=0):
        self.df = pd.read_csv(file_path, sep=None, header=None, names=['label', 'audio1', 'audio2'])
        if flag == 1:
            self.df = self.df.iloc[0:5000]
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio1 = self.df['audio1'].iloc[idx]
        audio1_list = str.split(audio1, '/')
        audio1_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/kb_data_clean_wav' + '/' + audio1_list[-5] + '/' + audio1_list[-4] + '/' + audio1_list[-3] + '/' + audio1_list[-2] + '/' + audio1_list[-1]
        input_audio1, org_sr = torchaudio.load(audio1_path)
        input_audio1 = torchaudio.functional.resample(input_audio1, orig_freq=org_sr, new_freq=16000)

        audio2 = self.df['audio2'].iloc[idx]
        audio2_list = str.split(audio2, '/')
        audio2_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/kb_data_clean_wav' + '/' + audio2_list[-5] + '/' + audio2_list[-4] + '/' + audio2_list[-3] + '/' + audio2_list[-2] + '/' + audio2_list[-1]
        input_audio2, org_sr = torchaudio.load(audio2_path)
        input_audio2 = torchaudio.functional.resample(input_audio2, orig_freq=org_sr, new_freq=16000)
        if self.transforms is not None:
                input_audio1 = self.transforms(input_audio1)
                input_audio2 = self.transforms(input_audio2)
        label = torch.tensor([self.df['label'].iloc[idx]])
        return label, input_audio1.squeeze(), input_audio2.squeeze()

def collate_fn(data):
    label, source1, source2 = zip(*data)
    source1_pad = pad_sequence(source1, batch_first=True)
    source2_pad = pad_sequence(source2, batch_first=True)
    label_pad = pad_sequence(label, batch_first=True)
    return label_pad, source1_pad, source2_pad


def train_epoch(model, valid_audio_loader, optimizer, lr_scheduler):

    model = model.cuda()
    model.train()
    model.freeze_feature_encoder()
    total_loss = 0
    num_batches = 0
    for idx, audio in enumerate(valid_audio_loader):
        label = audio[0].cuda()
        aud1 = audio[1].cuda()
        aud2 = audio[2].cuda()
        input = torch.cat((aud1, aud2), dim=1).cuda()
        preds = model(input_values=input, labels=label)
        loss = preds.loss
        total_loss = total_loss + loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        num_batches = idx + 1
    return total_loss/num_batches

def train(model, valid_audio_loader, optimizer, lr_scheduler, epochs):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        train_loss_epoch = train_epoch(model, valid_audio_loader, optimizer, lr_scheduler)
        print(f"Loss during training at epoch {epoch} : {train_loss_epoch}", flush=True)
        train_loss.append(train_loss_epoch)
        test_loss_epoch = test_epoch(model, test_audio_loader)
        print(f"Test Loss at epoch {epoch} : {test_loss_epoch}", flush=True)
        test_loss.append(test_loss_epoch)
        model.save_pretrained("/home/ubuntu/sapa/Speaker_Verification/models/wavlm_finetuned/")
    return train_loss, test_loss
    
        
def test_epoch(model, test_audio_loader):
    model = model.cuda()
    total_loss = 0
    num_batches = 0
    model.eval()
    for idx, audio in enumerate(test_audio_loader):
        label = audio[0].cuda()
        aud1 = audio[1].cuda()
        aud2 = audio[2].cuda()
        input = torch.cat((aud1, aud2), dim=1).cuda()
        with torch.no_grad():
            preds = model(input_values=input, labels=label)
        loss = preds.loss
        total_loss = total_loss + loss
        num_batches = idx + 1
    return total_loss/num_batches


if __name__ == "__main__":

    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
    model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    n_epochs = 10
    valid_file_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/meta_data/telugu/valid_data.txt'
    test_file_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/meta_data/telugu/test_data.txt'
    #train_epoch(model, valid_file_path)
    
    # flag=1 to consider just 5K samples
    valid_audio_dataset = kanbath_dataset(valid_file_path, flag=1)
    valid_audio_loader = DataLoader(valid_audio_dataset, collate_fn=collate_fn, batch_size=2)

    # flag=1 to consider just 5K samples
    test_audio_dataset = kanbath_dataset(test_file_path, flag=1)
    test_audio_loader = DataLoader(test_audio_dataset, collate_fn=collate_fn, batch_size=2)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = n_epochs * len(valid_audio_loader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    train_loss_all, test_loss_all = train(model, valid_audio_loader, optimizer, lr_scheduler, n_epochs)
    print("Model training Completed (5000 samples)")