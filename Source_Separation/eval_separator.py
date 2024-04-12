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
        source1 = read_audio(source_1_path).squeeze()
        source_2_path = self.df['source_2_path'].iloc[idx]
        source2 = read_audio(source_2_path).squeeze()
        mixture_path = self.df['mixture_path'].iloc[idx]
        mixture = read_audio(mixture_path).squeeze()
        return mixture, source1, source2

def collate_fn(data):
    mixture, source1, source2 = zip(*data)
    mixture_pad = pad_sequence(mixture, batch_first=True)
    source1_pad = pad_sequence(source1, batch_first=True)
    source2_pad = pad_sequence(source2, batch_first=True)
    return mixture_pad, source1_pad, source2_pad

def eval(model, file_path):
    audio_dataset = source_mix_dataset(file_path)
    
    generator = torch.Generator().manual_seed(42)
    train_audio_set, test_audio_set = random_split(audio_dataset, [0.7, 0.3], generator=generator)

    test_audio_loader = DataLoader(test_audio_set, collate_fn=collate_fn, batch_size=8)
    #test_audio_loader = DataLoader(test_audio_set, batch_size=1)


    si_snr_list = []
    sdr_list = []
    si_snr_list_mix = []
    sdr_list_mix = []
    for idx, audio in enumerate(test_audio_loader):
        si_snr = ScaleInvariantSignalNoiseRatio()
        sdr = SignalDistortionRatio()
        sep = model.separate_batch(audio[0])
        si_snr_list.append(si_snr(sep[:,:,0].cpu(), audio[1].cpu()).item())
        sdr_list.append(sdr(sep[:,:,0].cpu(), audio[1].cpu()).item())
        si_snr_list_mix.append(si_snr(audio[0].cpu(), audio[1].cpu()).item())
        sdr_list_mix.append(sdr(audio[0].cpu(), audio[1].cpu()).item())
    si_snr_avg = np.average(si_snr_list)
    sdr_avg = np.average(sdr_list)
    si_snr_avg_mix = np.average(si_snr_list_mix)
    sdr_avg_mix = np.average(sdr_list_mix)
    return si_snr_avg, sdr_avg, si_snr_avg_mix, sdr_avg_mix

if __name__ == "__main__":
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})
    print("Type of mixture : mix_clean (utterances only) :", flush=True)
    print("\n")
    print("Libri2Mix 8k sample rate and mode is min (the mixture ends when the shortest source ends) :", flush=True)
    print("\n", flush=True)
    si_snr_avg, sdr_avg, si_snr_avg_mix, sdr_avg_mix = eval(model, '/home/ubuntu/sapa/Source_Separation/dataset/Libri2Mix/wav8k/min/metadata/mixture_test_mix_clean.csv')
    print("Results for component1 of extracted output and source1 : ", flush=True)
    print("Scale-invariant signal-to-noise ratio improvement (SISNRi) on Test partition :", si_snr_avg, flush=True)
    print("Signal-to-distortion ratio improvement (SDRi) on Test partition :", sdr_avg, flush=True)
    print("\n", flush=True)
    print("Results for mixture and source1 : ", flush=True)
    print("Scale-invariant signal-to-noise ratio improvement (SISNRi) :", si_snr_avg_mix, flush=True)
    print("Signal-to-distortion ratio improvement (SDRi) :", sdr_avg_mix, flush=True)

    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)

    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr', run_opts={"device":"cuda"})
    print("Libri2Mix 8k sample rate and mode is max (the mixtures ends with the longest source) :", flush=True)
    print("\n", flush=True)
    si_snr_avg, sdr_avg, si_snr_avg_mix, sdr_avg_mix = eval(model, '/home/ubuntu/sapa/Source_Separation/dataset/Libri2Mix/wav8k/max/metadata/mixture_test_mix_clean.csv')
    print("Results for component1 of extracted output and source1 : ", flush=True)
    print("Scale-invariant signal-to-noise ratio improvement (SISNRi) on Test partition :", si_snr_avg, flush=True)
    print("Signal-to-distortion ratio improvement (SDRi) on Test partition :", sdr_avg, flush=True)
    print("\n", flush=True)
    print("Results for mixture and source1 : ", flush=True)
    print("Scale-invariant signal-to-noise ratio improvement (SISNRi) :", si_snr_avg_mix, flush=True)
    print("Signal-to-distortion ratio improvement (SDRi) :", sdr_avg_mix, flush=True)

