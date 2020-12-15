from src import data
import torch
# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition


import sys

if __name__ == "__main__":
    device = "cpu"

    data_dir = "../data/demo"  # a punctuation-free datafile
    demo_set = data.AutoPuncDataset(data_dir)

    print("Without punctuation:")
    unpunctuated_strings = data.get_punctuated_strings(demo_set, demo_set.raw_labels)
    for unpunc in unpunctuated_strings:
        print(unpunc)
    print()

    model = torch.load(sys.argv[1], map_location=device)
    predictions = []
    for speech, _ in demo_set.get_speeches():
        speech["tokens"] = speech["tokens"].to(device)
        speech["pros_feat"] = speech["pros_feat"].to(device)
        with torch.no_grad():
            predictions.append(model.predict(speech))

    print("\nAfter punctuation restoration:")
    for punctuated_string in data.get_punctuated_strings(demo_set, predictions):
        print(punctuated_string)
        print()
