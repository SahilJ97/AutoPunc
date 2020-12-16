from src import data
import torch
# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition


import sys

if __name__ == "__main__":
    model_file = sys.argv[1]
    print(f"Demonstrating {model_file}")
    device = "cpu"

    data_dir = "../data/demo"  # a punctuation-free datafile
    demo_set = data.AutoPuncDataset(data_dir)

    print("Without punctuation:")
    unpunctuated_strings = data.get_punctuated_strings(demo_set, demo_set.raw_labels)
    for unpunc in unpunctuated_strings:
        print(unpunc)
    print()

    model = torch.load(model_file, map_location=device)
    predictions = []
    for speech, _ in demo_set.get_speeches():  # get model predictions for each speech
        speech["tokens"] = speech["tokens"].to(device)
        speech["pros_feat"] = speech["pros_feat"].to(device)
        with torch.no_grad():
            predictions.append(model.predict(speech))

    print("\nAfter punctuation restoration:")
    for punctuated_string in data.get_punctuated_strings(demo_set, predictions):  # use predictions to perform APR
        print(punctuated_string)
        print()
