#!/bin/bash

# Running the script: sudo ./get_prosodic.sh <corpus_dir> <output_dir>
# IMPORTANT: Make sure the Gentle Docker container is running! Open a separate screen and run
# $ sudo docker run -P lowerquality/gentle
# You may need to change the value of GENTLE_PORT in forced_align.py; use
# $ docker ps
# to figure out where Docker is mapping the port at http://localhost:8765

CORPUS_DIR=$1  # path to NSC
OUTPUT_DIR=$2  # where TSV data files will be placed

PATH_TO_KALDI=/home/saj2163/kaldi-trunk  # may need to change

# Clean output directory
rm -rf "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR"

for part in 1 2
do
  common_base="$CORPUS_DIR"/PART"$part"/DATA/CHANNEL0
  # Data organization (relative to common_base):
  # WAV: WAVE/SPEAKER<SPEAKER_ID>/SESSION<SESSION #>/0<SPEAKER_ID><SESSION #><UTTERANCE_ID>.wav
  # TXT: SCRIPT/0<SPEAKER_ID><SESSION #>.TXT, and each TXT file indexes sentences by WAV basenames
  for speaker_zip in "$common_base"/WAVE/*  # should be speaker_zip. need to unzip.
  do
    unzip -q "$speaker_zip" -d "$common_base"/WAVE/
    speaker_dir=${speaker_zip/%.zip}
    speaker_id=$(echo "${speaker_dir/#$common_base}" | sed 's/[^0-9]*//g')
    audio_dir=$common_base/WAVE/SPEAKER$speaker_id
    echo Processing contents of $audio_dir ...
    for session in 0 1
    do
      speaker_session_transcript="$common_base"/SCRIPT/0"$speaker_id""$session".TXT
      ss_basename=$(basename "$speaker_session_transcript")
      speaker_session=${ss_basename/%.TXT}
      scp_file="$OUTPUT_DIR"/"$part"-"$speaker_session"-wav.scp
      python3 forced_align.py "$speaker_session_transcript" "$audio_dir" "$OUTPUT_DIR"/"$part"-"$speaker_id"-wav.scp
      "$PATH_TO_KALDI"/src/featbin/compute-mfcc-feats scp:"$scp_file" --config=confs/mfcc.conf
      "$PATH_TO_KALDI"/src/featbin/compute-kaldi-pitch-feats scp:"$scp_file" --config=confs/pitch.conf
    done
    rm -rf $speaker_dir
    exit 0  # just do one sub-loop iteration for now
  done
done
