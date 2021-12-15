import os, json
import argparse
from time import time
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.core.arrays.sparse import dtype
from pandas.core.frame import DataFrame
from tqdm import tqdm
from utils.kaldi_data import read_segments
from utils.clustering import cosine, hungarian
from kaldiio.matio import load_mat
from scipy.io import wavfile as wf
import math, GPUtil, time, pickle

from lab_master_graph import *

if __name__ == "__main__":
    
    np.seterr(divide='ignore',invalid='ignore')

    parser = argparse.ArgumentParser('online diarization')
    parser.add_argument('--stop-thres',     type=float, default=0.6)
    parser.add_argument('--seg-thres',      type=float, default=0.5)
    parser.add_argument('--spk-thres',      type=float, default=-1)
    parser.add_argument('--graph-thres',    type=float, default=0.3)
    parser.add_argument('--spklen',         type=float, default=6)
    parser.add_argument('--max-clsts',      type=int,   default=50)
    parser.add_argument('--method',         type=str,   default='chkpt-ahc')                     # ahc or chkpt-ahc
    parser.add_argument('--embd_scp',       type=str,   default="data/dihard3_dev/subsegments_data/embedding.scp")
    parser.add_argument('--segments',       type=str,   default="data/dihard3_dev/subsegments_data/segments")
    parser.add_argument('--output_path',    type=str,   default="./exp/dihard3_dev/rttm")
    parser.add_argument('--verbose',        action='store_true', default=False)                  # use this option when you want to see the output labels
    parser.add_argument('--verbose-time',   action='store_true', default=False)                  # use this option when you want to record the time

    args = parser.parse_args()

    if args.verbose_time:
        if not os.path.exists("./exp/times"):
            os.mkdir("./exp/times")

    # path for original recordings
    reco2wav_path = dict(x.split() for x in open("./data/dihard3_dev/wav.scp"))

    for f in args.embd_scp, args.segments:
        assert os.path.isfile(f), "NO SUCH FILE: %s" % f

    # read in segments
    utt2embd_path = dict(x.split() for x in open(args.embd_scp))
    reco2seg      = read_segments(args.segments)
    recos         = sorted(list(reco2seg.keys()))

    for idx, reco in enumerate(tqdm(recos)):

        if args.verbose_time:
            with open('./exp/times/{}.log'.format(reco), 'a+') as of:
                of.write("reading,ahc,reclustering,hungarian\n")

        st = time.time()

        lb = lab_master()
        labs       = []                                                                             # initialize final labels
        embds      = []                                                                             # initialize embeddings
        seg_lens   = []
        labs_hidden = []

        ''' segmentation and extract embeddings
        '''
        for utt, _, start, end in reco2seg[reco].values:

            seg_embd = load_mat(utt2embd_path[utt])
            embds.append(seg_embd)
            seg_lens.append(end-start)

            read_st = time.time()
            lb.read(seg_embd, start, end, args.seg_thres, args.graph_thres)
            read_ed = time.time()
            ahc_t, reclst_t = lb.update(args.stop_thres, args.spklen, args.max_clsts, method=args.method)
            labs_hidden = lb.labs

            ''' Hungarian Algorithm
                1. label alignment
                2. take advantage of the embeddings' information
            '''
            hungarian_st = time.time()
            labs = hungarian(labs, labs_hidden)
            hungarian_ed = time.time()

            if args.verbose:
                with open('./exp/output.log', 'a+') as of:
                    of.write(str(labs[-1])+' ')

            if args.verbose_time:
                with open('./exp/times/{}.log'.format(reco), 'a+') as of:
                    of.write("{},{},{},{}\n".format(read_ed-read_st, ahc_t, reclst_t, hungarian_ed-hungarian_st))

        if args.verbose:
            with open('./exp/output.log', 'a+') as of: of.write("\n")

        res = []
        for (utt, reco, start, end), lab in zip(reco2seg[reco].values, labs):
            if res == [] or res[-1]['end'] < start:
                res.append({'utt': utt, 'reco': reco, 'start': start, 'end': end, 'lab': lab})
            else:   # res is not empty and last_end >= start
                if res[-1]['lab'] == lab:
                    res[-1]['end'] = end
                else:
                    res[-1]['end'] = start = (res[-1]['end'] + start) / 2
                    res.append({'utt': utt, 'reco': reco, 'start': start, 'end': end, 'lab': lab})

        with open(args.output_path, 'a+') as f:
            fmt = "SPEAKER {0} {1} {2:7.3f} {3:7.3f} <NA> <NA> {4} <NA> <NA>\n"
            channel = 1
            for r in res:
                start = r['start'];  end = r['end'];  spk = r['lab']
                f.write(fmt.format(reco, channel, start, end - start, spk))

        ed = time.time()

        if args.verbose:
            with open('./exp/output.log', 'a+') as of:
                of.write("Record {0} finish! Takes {1:.3f}s in total.\n\n".format(reco, ed-st))
