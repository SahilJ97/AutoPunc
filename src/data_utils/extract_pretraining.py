# Script for preparing a pretraining datafile from a file provided for IWSLT12. The file is not paired with audio.
# Note that pretrain.data is not used in the training pipeline for this project.

if __name__ == "__main__":
    with open("../../data/iwslt/IWSLT12.TALK.train.en", "r") as in_file:
        with open("../../data/iwslt/pretrain.data", "w") as out_file:
            for line in in_file:
                for word in line.strip().split():
                    out_file.write(f"{word}\t[]\n")
