# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition
#
# Script for preparing a pretraining datafile from a file provided for IWSLT12. The file is not paired with audio.

if __name__ == "__main__":
    with open("../data/iwslt/IWSLT12.TALK.train.en", "r") as in_file:
        with open("../data/iwslt/train.data", "w") as out_file:
            for line in in_file:
                for word in line.strip().split():
                    out_file.write(f"{word}\t[]\n")
