#!/bin/bash

cd ../src || exit 1

# Train model on NSC data and tune it on IWSLT data, generating model-untuned.pt and model-tuned.pt
python3 ../src/train.py model.pt

# Evaluate untuned model
python3 ../src/eval.py model-untuned.pt ../data/nsc/test
python3 ../src/eval.py model-untuned.pt ../data/nsc/test --ignore-prosodic
python3 ../src/eval.py model-untuned.pt ../data/iwslt/tst2012_audio

# Evaluate tuned model
python3 ../src/eval.py model-tuned.pt ../data/iwslt/tst2012_audio
python3 ../src/eval.py model-tuned.pt ../data/iwslt/tst2012_audio --ignore-prosodic

cd ../scripts || exit 1
