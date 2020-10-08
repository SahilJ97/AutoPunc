from abc import ABC
import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.nn.modules import Module
from torch.utils.data import Dataset, DataLoader
from csv import DictReader


POSSIBLE_LABELS = ['.', ',', '?', None]


class AutoPuncDataset(Dataset):
    def __init__(self, csv_file, window_size, pretrained="roberta-base"):
        self.csv_file = csv_file
        self.window_size = window_size
        self.tokenizer = RobertaTokenizer(pretrained)
        self.sequence_features, self.labels = [], []
        self.load_data()

    def load_data(self):
        with open(self.csv_file, "r") as f:
            reader = DictReader(f)
            for row in reader:  # No! each input will be csv of (word, pros) pairs (one per line) for the entire
                # monologue. Zero pad on ends according to window size (can be done in __getitem__()).
                orig_text = row["transcription"]
                orig_prosodic_features = eval(row["prosodic_features"])
                i = -1
                tokens = []
                new_prosodic_features = []  # new_prosodic_features differs from orig_prosodic_features in that some \
                # entries may be replicated; some words in the transcription may be converted to 2+ tokens
                for word, pros in zip(orig_text.split(), orig_prosodic_features):
                    punc = None
                    if word[-1] in POSSIBLE_LABELS:
                        if len(word) > 1 and word[-2] != '.':  # Don't add '.' label if word ends with ellipsis
                            continue
                        punc = word[-1]
                        word = word[:-1]
                    new_tokens = self.tokenizer(word)['input_ids'][1:-1]
                    for token in new_tokens:
                        tokens.append(token)
                        new_prosodic_features.append(pros)
                        self.labels.append(POSSIBLE_LABELS.index(None))
                        i += 1
                    self.labels[-1] = POSSIBLE_LABELS.index(punc)
                self.sequence_features.append(
                    {"tokens": tokens, "prosody": new_prosodic_features}
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": self.sequence_features[idx],
            "label": self.labels[idx]
        }


class AutoPuncModel(Module, ABC):
    def __init__(self, window_size, pretrained_model="roberta-base"):
        super(AutoPuncModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained(pretrained_model)
        self.window_size = window_size

    def forward(self, x):
        return
