import torch
from transformers import RobertaTokenizer
from torch.utils.data import Dataset
import glob
import numpy as np
from typing import List

POSSIBLE_LABELS = ['.', ',', '?', None]


class AutoPuncDataset(Dataset):
    def __init__(
            self,
            data_dir,
            max_prosodic_seq_length=175,  # mean in train set is 167.6
            window_size=100,
            pretrained="roberta-base",
            n_pros_feat=4,
            ignore_prosodic=False
    ):
        self.data_files = glob.glob(f"{data_dir}/*.data")
        self.window_size = window_size
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained)
        self.max_seq_len = max_prosodic_seq_length
        self.n_pros_feat = n_pros_feat
        self.ignore_prosodic = ignore_prosodic
        self.data_map = []  # holds as list of (token, row_number) tuples for each speech
        self.raw_labels = []  # if data files are un-punctuated, labels will all be None
        self.index_data()

    def index_data(self):
        for data_file in self.data_files:
            print(f"Indexing {data_file}")
            self.data_map.append((data_file, []))
            with open(data_file, "r") as f:
                labels = []
                row_number = 0
                for row in f:
                    row = row.replace('"', '').replace("'", "").split("\t")
                    word = row[0].lower()
                    punc = None
                    if word == "":
                        continue
                    if word[-1] in POSSIBLE_LABELS:
                        if len(word) <= 1 or word[-2] != '.':  # don't add '.' label if word ends with ellipsis
                            punc = word[-1]
                            word = word[:-1]
                    new_tokens = self.tokenizer(word)['input_ids'][1:-1]
                    for token in new_tokens:
                        labels.append(POSSIBLE_LABELS.index(None))
                        self.data_map[-1][1].append((token, row_number))
                    labels[-1] = POSSIBLE_LABELS.index(punc)
                    row_number += 1
                self.raw_labels.append(labels)

    def __len__(self):
        return sum(
            len(speech[1])+1-self.window_size for speech in self.data_map
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
        zero = tuple((0. for i in range(self.n_pros_feat)))
        cropped = seq[max(0, len(seq)-self.max_seq_len):]  # trim the left side of the sequence
        padding = [zero for i in range(self.max_seq_len - len(seq))]
        return np.array(padding + cropped)

    def __getitem__(self, idx):
        speech_idx, offset = self.translate_index(idx)
        speech = self.data_map[speech_idx]
        return (
            self.get_input_window(speech, offset, offset + self.window_size),
            torch.tensor(self.raw_labels[speech_idx][offset: offset + self.window_size])
        )

    def get_input_window(self, speech, start, stop):
        tokens, pros_feat = [], []
        with open(speech[0], "r") as speech_file:
            lines = speech_file.readlines()
            for word, line_n in speech[1][start:stop]:
                tokens.append(word)
                pros_seq = []
                if not self.ignore_prosodic:
                    pros_seq = eval(
                        lines[line_n].split("\t")[1]
                    )
                pros_feat.append(pros_seq)
        formatted_pros_feat = np.stack([self.pad_and_crop_seq(seq) for seq in pros_feat])
        return {
            "tokens": torch.tensor(tokens),
            "pros_feat": torch.from_numpy(formatted_pros_feat)
        }

    def get_speeches(self):
        for speech, labels in zip(self.data_map, self.raw_labels):
            yield (
                self.get_input_window(speech, 0, len(speech[1])),
                torch.tensor(labels)
            )


def get_punctuated_strings(
        dataset: AutoPuncDataset,  # punctuation-free dataset
        predictions: List[torch.tensor]
):
    for data_fname, speech_map, speech_predictions in zip(dataset.data_files, dataset.data_map, predictions):
        s = ""
        with open(data_fname, "r") as data_file:
            token_index = 0
            for line in data_file:
                s += line.split("\t")[0]
                while token_index < len(speech_map[1]) - 1 and \
                        speech_map[1][token_index+1][1] == speech_map[1][token_index][1]:
                    token_index += 1
                pred = speech_predictions[token_index]
                if not isinstance(pred, int):
                    pred = pred.numpy().astype(float)
                    pred = np.argmax(pred)
                punc = POSSIBLE_LABELS[pred]
                if punc:
                    s += punc
                s += " "
                token_index += 1
        yield s
