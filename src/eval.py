from src.data import AutoPuncDataset, POSSIBLE_LABELS
from sys import argv
import torch
from torch.nn.functional import one_hot
from sklearn.metrics import precision_recall_fscore_support
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataset):
    all_preds, all_labels = [], []
    i = 0
    for speech, labels in dataset.get_speeches():
        print(f"Evaluating model for speech {i}...")
        i += 1
        speech["tokens"] = speech["tokens"].to(device)
        speech["pros_feat"] = speech["pros_feat"].to(device)
        labels = labels.to(device)

        all_preds.append(model.predict(speech))
        all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_labels = all_labels.numpy().astype(int)
    all_preds = all_preds.numpy().astype(int)
    all_preds = np.argmax(all_preds, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds)
    p_combined, r_combined, f_combined, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    return p, r, f, p_combined, r_combined, f_combined


if __name__ == "__main__":
    MODEL, TEST_DIR = argv[1], argv[2]
    device = "cpu"  # Use all CPU cores for evaluation on the test set
    NUM_PROCESSES = cpu_count()
    # load model and test set
    model = torch.load(MODEL, map_location=device)
    test_set = AutoPuncDataset(TEST_DIR)
    print("Evaluating model...")
    with torch.no_grad():
        results = evaluate(model, test_set)
    print("Results: ", results)

    model.share_memory()
    processes = []
    for rank in range(NUM_PROCESSES):  # fix: screenshotted error.
        p = mp.Process(target=evaluate, args=(model, test_set))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
