#!/bin/bash

# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition

# Start the Gentle forced aligner as a background process
docker run -P lowerquality/gentle &
sleep 10

# Extract NSC data to ../data/nsc/all_data/
bash extract_nsc_data.sh ~/nsc ../data/nsc/all_data

# Split NSC data into train, dev, and test sets
python3 split_data.py ../data/nsc/all_data

# Extract IWSLT 2012 test data to ../data/iwslt/tst2012/
python3 extract_iwslt_test.py ../data/iwslt/en-fr-test/IWSLT12.TED.MT.tst2012.en-fr.en.xml \
 ../data/iwslt/tst2012_audio ../data/iwslt/tst2012_audio

# Extract IWSLT 2012 training data to ../data/iwslt/train.data (one file)
python3 extract_iwslt_train.py
