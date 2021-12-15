import math
import numpy as np
import os 
import pandas as pd
import soundfile
from python_speech_features import logfbank


def read_rttm(rttm_path):
    usecols = [1, 3, 4, 7]
    names = ['reco', 'start', 'end', 'spk']
    dtype = {'start':float, 'end':float}
    rttms = pd.read_csv(rttm_path, 
        delim_whitespace=True, usecols=usecols, names=names, dtype=dtype)
    rttms['end'] += rttms['start']
    reco2rttm = {reco:rttm for reco, rttm in rttms.groupby('reco')}
    return reco2rttm

def read_segments(segments_path):
    names = ['utt', 'reco', 'start', 'end']
    dtype = {'start':float, 'end':float}
    segs = pd.read_csv(segments_path, 
        delim_whitespace=True, names=names, dtype=dtype)
    reco2seg = {reco:seg for reco, seg in segs.groupby('reco')}
    return reco2seg

def read_reco2dur(reco2dur_path):
    return {x.split()[0]:float(x.split()[1]) for x in open(reco2dur_path)}

def read_wav_scp(wav_scp):
    return dict(x.split() for x in open(wav_scp))

def read_spk2utt(spk2utt_path):
    return {x.split()[0]:x.split()[1:] for x in open(spk2utt_path)}

def read_utt2spk(utt2spk_path):
    return dict(x.split()for x in open(utt2spk_path))

def seg_to_vad(seg, dur, frm_len):
    nfrm = int(dur / frm_len)
    vad = np.zeros(nfrm, dtype=bool)
    for utt, reco, start, end in seg.values:
        s = round(start / frm_len)
        e = round(end / frm_len)
        vad[s:e] = 1
    return vad

def rttm_to_labels(rttm, nframe, frame_len=0.01):
    spks = sorted(list(set(rttm['spk'])))
    nspk = len(spks)
    spk2int = {spk:i for i, spk in enumerate(spks)}
    labels = np.zeros((nframe, nspk), dtype=bool)
    for reco, start, end, spk in rttm.values:
        s = round(start / frame_len)
        e = round(end / frame_len)
        labels[s:e, spk2int[spk]] = 1
    return labels


class KaldiData():
    def __init__(self, data, fbank_kwargs=None, cmn=True, subsample=8):
        self.reco2seg = read_segments(os.path.join(data, 'segments'))
        self.reco2wav_path = read_wav_scp(os.path.join(data, 'wav.scp'))
        self.recos = sorted(list(self.reco2wav_path.keys()))
        self.len = len(self.recos)
        self.fbank_kwargs = fbank_kwargs or {
            "winlen": 0.025,
            "winstep": 0.01,
            "nfilt": 64,
            "nfft": 512,
            "lowfreq": 0,
            "highfreq": None,
            "preemph": 0.97,
            "winfunc": np.hamming
        }
        self.cmn = cmn
        self.subsample = subsample

    def __getitem__(self, index):
        reco = self.recos[index]
        try:
            seg = self.reco2seg[reco]
        except:
            return "nonspeech", None, None

        wav_path = self.reco2wav_path[reco] 
        y, sr = soundfile.read(wav_path)
        min_unit = self.subsample * self.fbank_kwargs['winstep']
        dur = len(y) / sr
        dur = math.ceil(dur / min_unit) * min_unit

        vad = seg_to_vad(seg, dur, self.fbank_kwargs['winstep'])

        nfrm = len(vad)
        nlen  = int(sr * self.fbank_kwargs['winlen'])
        nstep = int(sr * self.fbank_kwargs['winstep'])
        N = nstep * (nfrm - 1) + nlen
        if N < len(y):
            y = y[:N]
        else:
            y = np.pad(y, (0, N - len(y)), mode='constant')

        feat = logfbank(y, sr, **self.fbank_kwargs)
        if self.cmn:
            feat -= feat.mean(axis=0, keepdims=True)
        return reco, feat, vad[self.subsample//2::self.subsample]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    data = 'data/dihard_2019_dev'
    kaldi_loader = KaldiData(data)
    for reco, feat, vad in kaldi_loader:
        print(reco, feat.shape, vad.shape, vad.sum())
