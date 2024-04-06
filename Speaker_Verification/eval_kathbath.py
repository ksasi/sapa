
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
import librosa
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve



def compute_eer(labels, scores):
    """
    Ref : https://yangcha.github.io/EER-ROC/
    https://github.com/albanie/pytorch-benchmarks/blob/master/lfw_eval.py   
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


from transformers import AutoFeatureExtractor, UniSpeechSatForXVector, WavLMForXVector, Wav2Vec2ForXVector

def compute_similarity(model, audio1, audio2):
    """
    Function that computers similarity given model type and audio files
    """
    if model == 'XLSR-Wav2Vec2':
        # XLSR-Wav2Vec2
        # https://huggingface.co/docs/transformers/en/model_doc/xlsr_wav2vec2
        feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    elif model == 'UniSpeech-SAT':
        # UniSpeech-SAT
        # https://huggingface.co/docs/transformers/model_doc/unispeech-sat#usage-tips
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
        model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
    elif model == 'WavLM-Base':
        feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    audio1_list = str.split(audio1, '/')
    audio1_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/kb_data_clean_wav' + '/' + audio1_list[-5] + '/' + audio1_list[-4] + '/' + audio1_list[-3] + '/' + audio1_list[-2] + '/' + audio1_list[-1]
    input_audio1, sampling_rate = librosa.load(audio1_path,  sr=16000)

    audio2_list = str.split(audio2, '/')
    audio2_path = '/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/kb_data_clean_wav' + '/' + audio2_list[-5] + '/' + audio2_list[-4] + '/' + audio2_list[-3] + '/' + audio2_list[-2] + '/' + audio2_list[-1]    
    input_audio2, sampling_rate = librosa.load(audio2_path,  sr=16000)
    inputs = feature_extractor([input_audio1, input_audio2], sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(embeddings[0], embeddings[1])
    return similarity.cpu().item()

def main():
    """
    Main function to computer EER for all 3 types of models
    """
    df = pd.read_csv('/home/ubuntu/sapa/Speaker_Verification/dataset/Kathbath/meta_data/telugu/test_data.txt', sep=None, header=None, names=['label', 'audio1', 'audio2'])
    df = df.iloc[0:5000]
    

    labels = [x for x in df['label']]
    scores_XLSR_Wav2Vec2 = [compute_similarity('XLSR-Wav2Vec2', x, y) for x, y in zip(df['audio1'], df['audio2'])]
    scores_UniSpeech_SAT = [compute_similarity('UniSpeech-SAT', x, y) for x, y in zip(df['audio1'], df['audio2'])]
    scores_WavLM_Base = [compute_similarity('WavLM-Base', x, y) for x, y in zip(df['audio1'], df['audio2'])]

    print("EER for XLSR-Wav2Vec2 model on Kathbath Telugu Test dataset : ", compute_eer(labels, scores_XLSR_Wav2Vec2)[0], flush=True)

    print("EER for UniSpeech-SAT model on Kathbath Telugu Test dataset : ", compute_eer(labels, scores_UniSpeech_SAT)[0], flush=True)

    print("EER for WavLM-Base  model on Kathbath Telugu Test dataset : ", compute_eer(labels, scores_WavLM_Base)[0], flush=True)


if __name__ == "__main__":
    main()