import kaldi_io
from sys import argv
from scipy.signal import resample
import os
import numpy as np

if __name__ == "__main__":
    transcript_file, energy_scp, pitch_scp, output_file = argv[1], argv[2], argv[3], argv[4]
    if os.path.exists(output_file):
        print(f"Output file {output_file} exists. Exiting.")
        exit()

    # Get prosodic sequences
    energy_data = kaldi_io.read_mat_scp(energy_scp)
    pitch_data = kaldi_io.read_mat_scp(pitch_scp)
    prosodic_sequences = {}
    for (energy_key, energy_mat), (pitch_key, pitch_mat) in zip(energy_data, pitch_data):
        if energy_key != pitch_key:
            print("Keys differ! Exiting.")
            exit()
        if len(energy_mat) != len(pitch_mat):
            print("Resampling energy sequence.")
            energy_mat = resample(energy_mat, len(pitch_mat), axis=1)
        prosodic_seq = [tuple(np.concatenate((e, p))) for e, p in zip(energy_mat, pitch_mat)]
        prosodic_sequences[energy_key] = prosodic_seq

    # Generate data file
    with open(transcript_file, "r", encoding='utf-8-sig') as transcript_file:
        transcript = transcript_file.read()
        words = transcript.strip().split()
        for i in range(len(words)):
            word_key = f"{words[i]}.{i}"
            prosodic = prosodic_sequences.setdefault(word_key, [])
            with open(output_file, "a+") as out_file:
                out_file.write(f"{words[i]}\t{prosodic}\n")
