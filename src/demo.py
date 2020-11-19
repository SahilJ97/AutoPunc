from src import data
import torch

if __name__ == "__main__":
    device = "cpu"

    data_dir = "../data/demo/"  # a punctuation-free datafile
    model = torch.load("model.pt", map_location=device)
    demo_set = data.AutoPuncDataset(data_dir)

    predictions = []
    for speech, _ in demo_set.get_speeches():
        speech["tokens"] = speech["tokens"].to(device)
        speech["pros_feat"] = speech["pros_feat"].to(device)

        predictions.append(model.predict(speech))

    for punctuated_string in data.get_punctuated_strings(demo_set, predictions):
        print(punctuated_string)
        print()
