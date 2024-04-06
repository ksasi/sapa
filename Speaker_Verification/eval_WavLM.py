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

def compute_eer(labels, scores):
    """
    Ref : https://yangcha.github.io/EER-ROC/
    https://github.com/albanie/pytorch-benchmarks/blob/master/lfw_eval.py   
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


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


def compute_similarity(model, feature_extractor, test_audio_loader):
    labels = []
    scores = []
    sampling_rate=16000
    for idx, audio in enumerate(test_audio_loader):
        inputs = feature_extractor([torch.flatten(audio[1]).numpy(), torch.flatten(audio[2]).numpy()], sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
        scores.append(similarity.cpu().item())
        labels.append(audio[0].cpu().item())
    return scores, labels

if __name__ == "__main__":

    test_file_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/meta_data/telugu/test_data.txt'

    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')

    # flag=1 to consider just 5K samples
    test_audio_dataset = kanbath_dataset(test_file_path, flag=1)

    ## TODO : Batch Size cannot be more than 1, other batch sizes causes compute_similarity fail
    test_audio_loader = DataLoader(test_audio_dataset, collate_fn=collate_fn, batch_size=1)

    model_finetuned = WavLMForXVector.from_pretrained('/home/ubuntu/sapa/Speaker_Verification/models/wavlm_finetuned')
    scores, labels = compute_similarity(model_finetuned, feature_extractor, test_audio_loader)
    eer, thresh = compute_eer(labels, scores)
    print(f"EER for WavLM-Base  model after fine-tuning on Kathbath Telugu Valid dataset : {eer}", flush=True)