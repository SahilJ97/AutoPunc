#!/bin/bash

# Running the script: sudo bash extract_data.sh <corpus_dir> <output_dir>
# IMPORTANT: Make sure the Gentle Docker container is running! Open a separate screen and run
# $ sudo docker run -P lowerquality/gentle
# You may need to change the value of GENTLE_PORT in forced_align.py; use
# $ docker ps
# to figure out where Docker is mapping the port at http://localhost:8765

CORPUS_DIR=$1  # path to NSC
OUTPUT_DIR=$2  # where TSV data files will be placed

PATH_TO_KALDI=/home/saj2163/kaldi-trunk  # change according to installation location
pip3 install -r ../../requirements.txt

common_base="$CORPUS_DIR"/PART2/DATA/CHANNEL0
# Data organization in corpus (relative to common_base):
# audio: WAVE/SPEAKER<SPEAKER_ID>/SESSION<SESSION #>/0<SPEAKER_ID><SESSION #><UTTERANCE_ID>.flac
# text: SCRIPT/0<SPEAKER_ID><SESSION #>.TXT, and each TXT file indexes sentences by .flac basenames
for speaker_dir in "$common_base"/WAVE/*; do
  speaker_id=$(echo "${speaker_dir/#$common_base/}" | sed 's/[^0-9]*//g')
  audio_dir=$common_base/WAVE/SPEAKER$speaker_id
  echo "Processing contents of $audio_dir ..."
  for session in 0 1; do
    speaker_session_transcript="$common_base"/SCRIPT/0"$speaker_id""$session".TXT
    if [ ! -d "$audio_dir"/SESSION"$session" ]; then
      continue
    fi
    ss_basename=$(basename "$speaker_session_transcript")
    speaker_session=${ss_basename/%.TXT/}
    wav_scp_file="$OUTPUT_DIR"/2-"$speaker_session"-wav.scp

    if [ ! -f "$wav_scp_file" ]; then
      echo "Aligning audio..."
      python3 forced_align.py "$speaker_session_transcript" "$audio_dir" "$wav_scp_file" &> /dev/null
    fi


    if [ ! -f 2-"$speaker_session".data ]; then
      echo "Computing pitch and with deltas..."
      "$PATH_TO_KALDI"/src/featbin/compute-and-process-kaldi-pitch-feats \
        --config=confs/pitch.conf scp:"$wav_scp_file" ark,scp:pitch.ark,pitch.scp

      echo "Computing energy..."
      "$PATH_TO_KALDI"/src/featbin/compute-mfcc-feats \
        --config=confs/mfcc.conf scp:"$wav_scp_file" ark,scp:energy.ark,energy.scp

      echo "Generating formatted data file..."
      KALDI_ROOT=$PATH_TO_KALDI python3 format_session_data.py \
        "$speaker_session_transcript" energy.scp pitch.scp "$OUTPUT_DIR"/2-"$speaker_session".data
    fi

    rm energy.scp energy.ark pitch.scp pitch.ark
  done
done

python3 split_data.py "$OUTPUT_DIR"
