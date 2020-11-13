from src.data import AutoPuncDataset, POSSIBLE_LABELS
from sys import argv
import torch
from torch.nn.functional import binary_cross_entropy, one_hot
from sklearn.metrics import precision_recall_fscore_support


def evaluate(model, dataset):
    all_preds, all_labels = [], []
    for speech, labels in dataset.get_speeches():
        all_preds.append(model.predict(speech))
        all_labels.append(one_hot(labels, num_classes=len(POSSIBLE_LABELS)).float())
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = all_preds.numpy()
    all_labels = all_labels.numpy()
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, labels=POSSIBLE_LABELS)
    p_combined, r_combined, f_combined, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    return p, r, f, p_combined, r_combined, f_combined


if __name__ == "__main__":
    MODEL, TEST_DIR = argv[1], argv[2]
    # load model and test set
    model = torch.load(MODEL)
    test_set = AutoPuncDataset(TEST_DIR)
    print("Evaluating model...")
    results = evaluate(model, test_set)
    print("Results: ", results)
