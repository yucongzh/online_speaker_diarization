# online_speaker_diarization

[Low-Latency Online Speaker Diarization with Graph-based Label Generation](https://www.isca-speech.org/archive/pdfs/odyssey_2022/zhang22_odyssey.pdf)  
by Yucong Zhang, Qinjian Lin, Weiqing Wang, Lin Yang, Xuyang Wang, Junjie Wang, Ming Li

> This paper introduces an online speaker diarization sys-tem that can handle long-time audio with low latency. We enable Agglomerative Hierarchy Clustering (AHC) to work in an online fashion by introducing a label matching algorithm. This algorithm solves the inconsistency between output labels and hidden labels that are generated each turn. To ensure the low latency in the online setting, we introduce a variant of AHC, namely chkpt-AHC, to cluster the speakers. In addition, we propose a speaker embedding graph to exploit a graph-based re-clustering method, further improving the performance. In the experiment, we evaluate our systems on both DIHARD3 and VoxConverse datasets. The experimental results show that our proposed online systems have better performance than our baseline online system and have comparable performance to our offline systems. We find out that the framework combining the chkpt-AHC method and the label matching algorithm works well in the online setting. Moreover, the chkpt-AHC method greatly reduces the time cost, while the graph-based re-clustering method helps improve the performance.

## Citation
```bitex
@inproceedings{zhang22_odyssey,
  author={Yucong Zhang and Qinjian Lin and Weiqing Wang and Lin Yang and Xuyang Wang and Junjie Wang and Ming Li},
  title={{Low-Latency Online Speaker Diarization with Graph-Based Label Generation}},
  year=2022,
  booktitle={Proc. The Speaker and Language Recognition Workshop (Odyssey 2022)},
  pages={162--169},
  doi={10.21437/Odyssey.2022-23}
}
```

## Installation
1. Create environment:
```
conda create -n diarization python=3.7
```
2. Install Python packages
```
pip install -r requirements.txt
```

## Usage
> *You can obtain the dihard3 data according to the instructions in [dihard3_baseline](https://github.com/dihardchallenge/dihard3_baseline#expected-results)*

1. Data preparation  
Replace the `{ROOTDIR}` in `data/dihard3_dev/wav.scp` with your absolute path to this repository. Use `run.sh` to prepare the audio segments. All the audio segments are saved under `subsegments/` folder, located under `data/dihard3_dev/` folder. `dihard3_dev/` folder will be created under `exp/` folder as well.

```
DH_DEV_0001 {ROOTDIR}/online_speaker_diarization/data/dihard3_dev/example_wav/DH_DEV_0001.wav
```

2. Generate speaker labels in an online fashion  
The `--verbose` and `--verbose-time` options enable you to record the labels and time, which are saved as `output.log` and `times/` respectively under `exp/` folder.

```
python online_diarization.py --verbose --verbose-time
```

3. Computing DERs
```
dscore/score.py -r data/dihard3_dev/rttm -s exp/dihard3_dev/rttm
```
or  
```
dscore/scorelib/md-eval.pl -r data/dihard3_dev/rttm -s exp/dihard3_dev/rttm
```

## Acknowledgements
We borrow some codes from kaldi.
