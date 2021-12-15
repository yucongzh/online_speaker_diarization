import torch
from scipy.io import wavfile as wf
from python_speech_features import sigproc

from modules.model_spk import ResNet34StatsPool
from dataset.feats import logFbankCal
import numpy as np


def preprocess_signal(signal, preemph=0.97):
    if np.std(signal) == 0:
        return signal
    else:
        # signal = signal / np.abs(signal).max()
        signal = (signal - np.mean(signal)) / np.std(signal)
        return sigproc.preemphasis(signal, preemph).astype('float32')


class SIDInfer:
    def __init__(self, resume, device='cuda'):
        self.device = torch.device(device)
        self.featCal = logFbankCal(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80).to(self.device)
        self.featCal.eval()

        self.model = ResNet34StatsPool(in_planes=34, embedding_size=128)
        self.model.load_state_dict(torch.load(resume, map_location=self.device)['model'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_wav(self, wav_path):
        sr, y = wf.read(reco2wav_path[reco])
        return y, sr

    def __call__(self, y, sr):
        y = preprocess_signal(y)
        assert sr == 16000 and y.ndim == 1
        with torch.no_grad():
            y = torch.from_numpy(y).unsqueeze(0).to(self.device)
            feat = self.featCal(y).squeeze(1).transpose(1, 2)
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd


import pandas as pd
def read_segments(segments_path):
    names = ['utt', 'reco', 'start', 'end']
    dtype = {'start':float, 'end':float}
    segs = pd.read_csv(segments_path,
        delim_whitespace=True, names=names, dtype=dtype)
    reco2seg = {reco:seg for reco, seg in segs.groupby('reco')}
    return reco2seg


if __name__ == '__main__':
    import os
    import math
    import GPUtil
    import kaldiio
    import argparse
    import numpy as np
    from tqdm import tqdm

    parser = argparse.ArgumentParser('')
    parser.add_argument('-r', '--resume', 
        default='DKU_SID/vox2_ResNet34StatsPool-32-128_ArcFace-64-0.2_model_81.pkl', type=str)
    parser.add_argument('wav_scp')
    parser.add_argument('segments')
    parser.add_argument('embd_wdir')
    args = parser.parse_args()

    for f in args.wav_scp, args.segments:
        assert os.path.isfile(f), "NO SUCH FILE: %s" % f

    gpu = GPUtil.getAvailable(limit=1, maxMemory=0.01, maxLoad=0.01)[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print("Use GPU: %d" % gpu)

    sid_infer = SIDInfer(args.resume, device='cuda')

    reco2wav_path = dict(x.split() for x in open(args.wav_scp))
    recos = sorted(list(reco2wav_path.keys()))
    reco2seg = read_segments(args.segments)

    wfile = 'ark,scp:{0}/embedding.ark,{0}/embedding.scp'.format(args.embd_wdir)
    writer = kaldiio.WriteHelper(wfile)
    for reco in tqdm(recos, ncols=50):
        y, sr = sid_infer.load_wav(reco2wav_path[reco])
        for utt, reco, start, end in reco2seg[reco].values:
            s = int(start * sr)
            e = int(end * sr)
            input = y[s:e]
            if len(input) < 8000:
                N = math.ceil(8000 / len(input))
                input = np.concatenate([input] * N)[:8000]
            embd = sid_infer(input, sr)
            writer(utt, embd)
    writer.close()
