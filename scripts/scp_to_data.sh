#!/bin/bash

# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition


SCP_FILE=$1
TRANSCRIPT=$2
OUTPUT_FILE=$3

PATH_TO_KALDI=/home/saj2163/kaldi-trunk  # change according to installation location

echo "Computing pitch with deltas..."
"$PATH_TO_KALDI"/src/featbin/compute-and-process-kaldi-pitch-feats \
  --config=confs/pitch.conf scp:"$SCP_FILE" ark,scp:pitch.ark,pitch.scp

echo "Computing energy..."
"$PATH_TO_KALDI"/src/featbin/compute-mfcc-feats \
  --config=confs/mfcc.conf scp:"$SCP_FILE" ark,scp:energy.ark,energy.scp

echo "Generating formatted data file..."
KALDI_ROOT=$PATH_TO_KALDI python3 write_datafile.py "$TRANSCRIPT" energy.scp pitch.scp "$OUTPUT_FILE"

rm energy.scp energy.ark pitch.scp pitch.ark