# use Montreal Forced Aligner to extract prosodic features and Bert to generate token embeddings
# https://montreal-forced-aligner.readthedocs.io/en/latest/example.html
# just use Gentle!!! https://github.com/lowerquality/gentle


if __name__ == "__main__":
    # Preprocess pretraining data (text only)
    with open("../data/raw_data/IWSLT12.TALK.train.en", "r") as in_f:
        with open("../data/pretraining/pretrain.csv", "w+") as out_f:
            for line in in_f:
                for word in line.strip().split():
                    out_f.write(word.lower() + "\n")
