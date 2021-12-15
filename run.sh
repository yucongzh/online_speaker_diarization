#!/bin/bash
set -e

winlen=1.0      # window length of uniform segmentation
winshift=0.5    # window step of uniform segmentation

stage=0
data=data/dihard3_dev
exp=exp/dihard3_dev
subdata=$data/subsegments_data
. utils/parse_options.sh || exit 1

mkdir -p $exp $subdata

if [ $stage -le 0 ]; then
  echo "Stage 0: Generate segment-wise embeddings with ${winlen}s length and ${winshift}s step"
  # Convert segments into split segments with maximum duration of winlen.
  awk '{print $1,$2}' $data/segments > $data/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
  utils/make_subsegments_data.sh --winlen $winlen --winshift $winshift \
    --min-segment 0.5 $data $subdata

  python DKU_SID/sid_infer.py $data/wav.scp $subdata/segments $subdata
fi
