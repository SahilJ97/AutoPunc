import torch
from transformers import RobertaTokenizer
from torch.utils.data import Dataset
from csv import reader
import glob
import numpy as np

POSSIBLE_LABELS = ['.', ',', '?', None]


class AutoPuncDataset(Dataset):
    def __init__(self, data_dir, max_prosodic_seq_length=500, window_size=100, pretrained="roberta-base"):
        self.csv_files = glob.glob(f"{data_dir}/*.data")
        self.window_size = window_size
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained)
        self.data = []  # each item in "data" corresponds to an entire speech.
        self.raw_labels = []
        self.load_data()
        self.max_seq_len = max_prosodic_seq_length

    def load_data(self):
        for csv_file in self.csv_files:  # wait...want to normalize???
            with open(csv_file, "r") as f:
                rows = reader(f, delimiter="\t")
                i = -1
                tokens = []
                prosodic_features = []
                labels = []
                for row in rows:  # row format: "word x1 x2 ...", where each xi is a tuple of prosodic features
                    word = row[0].lower()
                    prosodic = eval(row[1])
                    punc = None
                    if word[-1] in POSSIBLE_LABELS:
                        if len(word) <= 1 or word[-2] != '.':  # Don't add '.' label if word ends with ellipsis
                            punc = word[-1]
                            word = word[:-1]
                    new_tokens = self.tokenizer(word)['input_ids'][1:-1]
                    for token in new_tokens:
                        tokens.append(token)
                        prosodic_features.append(prosodic)
                        labels.append(POSSIBLE_LABELS.index(None))
                        i += 1
                    labels[-1] = POSSIBLE_LABELS.index(punc)
                self.data.append(
                    {
                        "tokens": tokens,
                        "prosodic_features": prosodic_features
                    }
                )
                self.raw_labels.append(labels)

    def __len__(self):
        return sum(
            len(speech["tokens"])+1-self.window_size for speech in self.data
        )

    def translate_index(self, idx):
        offset = idx
        for speech_idx in range(len(self.raw_labels)):
            speech = self.raw_labels[speech_idx]
            n_windows = len(speech) + 1 - self.window_size
            if offset - n_windows >= 0:
                offset -= n_windows
                continue
            return speech_idx, offset

    def pad_and_crop_seq(self, seq):
        zero = tuple((0. for feature in seq[0]))
        cropped = seq[max(0, len(seq)-self.max_seq_len):]  # trim the left side of the sequence
        padding = [zero for i in range(self.max_seq_len - len(seq))]
        return np.array(padding + cropped)

    def __getitem__(self, idx):
        speech_idx, offset = self.translate_index(idx)
        speech = self.data[speech_idx]
        tokens = speech["tokens"][offset: offset + self.window_size]
        pros_feat = speech["prosodic_features"][offset: offset + self.window_size]
        formatted_pros_feat = np.stack([self.pad_and_crop_seq(seq) for seq in pros_feat])
        return (
            {
                "tokens": torch.tensor(tokens),
                "pros_feat": torch.from_numpy(formatted_pros_feat)
            },
            torch.tensor(self.raw_labels[speech_idx][offset: offset + self.window_size])
        )

    def get_speeches(self):
        for speech, labels in zip(self.data, self.raw_labels):
            formatted_pros_feat = np.array([self.pad_and_crop_seq(seq) for seq in speech["prosodic_features"]])
            yield (
                {
                    "tokens": torch.tensor(speech["tokens"]),
                    "pros_feat": torch.from_numpy(formatted_pros_feat)
                },
                torch.tensor(labels)
            )
