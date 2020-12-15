# Written by Sahil Jayaram (saj2163) for COMS 6998 (Topics in Computer Science): Fundamentals of Speech Recognition


from src.data import AutoPuncDataset, POSSIBLE_LABELS
from sys import argv
import torch
from sklearn.metrics import precision_recall_fscore_support
from multiprocessing import cpu_count
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataset):
    all_preds, all_labels = [], []
    i = 0
    for speech, labels in dataset.get_speeches():
        print(f"Evaluating model for speech {i}...")
        i += 1
        speech["tokens"] = speech["tokens"].to(DEVICE)
        speech["pros_feat"] = speech["pros_feat"].to(DEVICE)
        labels = labels.to(DEVICE)

        all_preds.append(model.predict(speech))
        all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_labels = all_labels.numpy().astype(int)
    all_preds = all_preds.numpy().astype(float)  # just changed from int after batch 1 evaluation...
    all_preds = np.argmax(all_preds, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds)
    p_combined, r_combined, f_combined, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    # all the same! should I be using macro???
    return p, r, f, p_combined, r_combined, f_combined


if __name__ == "__main__":
    MODEL, TEST_DIR = argv[1], argv[2]
    IGNORE_PROSODIC = False
    if "--ignore-prosodic" in argv:
        IGNORE_PROSODIC = True
    DEVICE = "cpu"  # Use all CPU cores for evaluation on the test set
    NUM_PROCESSES = cpu_count()
    # load model and test set
    test_set = AutoPuncDataset(TEST_DIR, ignore_prosodic=IGNORE_PROSODIC)
    model = torch.load(MODEL, map_location=DEVICE)
    print("Evaluating model...")
    with torch.no_grad():
        results = evaluate(model, test_set)
    print(results)
    p, r, f, p_combined, r_combined, f_combined = results
    print("p, r, f, p_combined, r_combined, f_combined: ", p, r, f, p_combined, r_combined, f_combined)
