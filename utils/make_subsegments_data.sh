# Generate segments, spk2utt, utt2spk, wav.scp file in <data>/subsegments_data directory.
# Files above are after the sliding window.

#!/bin/bash 
set -e 

winlen=1.5
winshift=0.75
min_segment=0.5
. utils/parse_options.sh || exit 1;
data=$1
subdata=$2

[ ! -f $data/segments ] && echo "No such directory: $data/segments" && exit 1;

utils/get_uniform_subsegments.py --max-segment=$winlen \
  --overlap-duration=$(perl -e "print $winlen-$winshift") \
  --max-remaining-duration=$min_segment --constant-duration=True \
  $data/segments > $data/subsegments

mkdir -p $subdata
utils/subsegment_data_dir.sh $data $data/subsegments $subdata
rm $data/subsegments
