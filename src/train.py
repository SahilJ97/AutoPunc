from src.model import AutoPuncModel
from src.data import AutoPuncDataset, POSSIBLE_LABELS
from src.eval import evaluate
from sys import argv
import torch
from torch_optimizer import RAdam, Lookahead
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot

OUTPUT_FILE = argv[1]


def train(model, dataset, optimizer, batch_size=1000, num_epochs=9, loss_fn=binary_cross_entropy, val_set=None):
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = one_hot(labels, num_classes=len(POSSIBLE_LABELS)).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics every 10 mini-batches
            running_loss += loss.item()
            if i % 1 == 0:  # temporarily 10 -> 1, 9 -> 0
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        epoch_losses.append(running_loss)
        if running_loss == max(epoch_losses):
            print("Saving model...")
            torch.save(model, OUTPUT_FILE)
            print("Model saved.")

        if val_set:
            print("Validating...")
            p, r, f, p_combined, r_combined, f_combined = evaluate(model, val_set)
            print("p, r, f, p_combined, r_combined, f_combined: ", p, r, f, p_combined, r_combined, f_combined)

    print("Finished training. Epoch losses: ", epoch_losses)


if __name__ == "__main__":
    model = AutoPuncModel()
    print("Loading dataset...")
    pretrain_set = AutoPuncDataset("../data/train")
    dev_set = AutoPuncDataset("../data/dev")
    print("Training model...")
    radam = RAdam(model.parameters(), betas=(.9, .999), lr=1e-5, eps=1e-8)
    lookahead_optimizer = Lookahead(radam, k=6, alpha=0.5)
    train(model, pretrain_set, lookahead_optimizer, val_set=dev_set)
