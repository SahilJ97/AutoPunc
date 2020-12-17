#!/bin/bash

# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition

cd ../src || exit 1
export PYTHONPATH=$PYTHONPATH:..

# Train model on NSC data and tune it on IWSLT data, generating model-untuned.pt and model-tuned.pt
python3 ../src/train.py model.pt

# Evaluate untuned model
python3 ../src/eval.py model-untuned.pt ../data/nsc/test
python3 ../src/eval.py model-untuned.pt ../data/nsc/test --ignore-prosodic
python3 ../src/eval.py model-untuned.pt ../data/iwslt/tst2012_audio

# Evaluate tuned model
python3 ../src/eval.py model-tuned.pt ../data/iwslt/tst2012_audio
python3 ../src/eval.py model-tuned.pt ../data/iwslt/tst2012_audio --ignore-prosodic

# Demonstrate end-to-end system
python3 ../src/demo.py model-untuned.pt
python3 ../src/demo.py model-tuned.pt

cd ../scripts || exit 1