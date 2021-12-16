# online_speaker_diarization

[Online Speaker Diarization with Graph-based Label Generation](https://arxiv.org/abs/2111.13803)  
by Yucong Zhang, Qinjian Lin, Weiqing Wang, Lin Yang, Xuyang Wang, Junjie Wang, Ming Li

> This paper introduces an online speaker diarization system that can handle long-time audio with low latency. First, a new variant of ag-glomerative hierarchy clustering is built to cluster the speakers in an online fashion. Then, a speaker embedding graph is proposed. We use this graph to exploit a graph-based reclustering method to fur-ther improve the performance. Finally, a label matching algorithm is introduced to generate consistent speaker labels, and we evalu-ate our system on both DIHARD3 and VoxConverse datasets, which contain long audios with various kinds of scenarios. The experi-mental results show that our online diarization system outperforms the baseline ofﬂine system and has comparable performance to our ofﬂine system.

## Citation
```bitex
@article{zhang2021online,
  title={Online Speaker Diarization with Graph-based Label Generation},
  author={Zhang, Yucong and Lin, Qinjian and Wang, Weiqing and Yang, Lin and Wang, Xuyang and Wang, Junjie and Li, Ming},
  journal={arXiv preprint arXiv:2111.13803},
  year={2021}
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
> *Here we provide 2 audios from DIHARD3 dataset as an example*

1. Data preparation  
Replace the `{ROOTDIR}` in `data/dihard3_dev/wav.scp` with your absolute path to this repository. Use `run.sh` to prepare the audio segments. All the audio segments are saved under `subsegments/` folder, located under `data/dihard3_dev/` folder. `dihard3_dev/` folder will be created under `exp/` folder as well.

```
DH_DEV_0001 {ROOTDIR}/online_diarization/data/dihard3_dev/example_wav/DH_DEV_0001.wav
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