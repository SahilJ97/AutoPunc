from src.model import AutoPuncModel
from src.data import AutoPuncDataset, POSSIBLE_LABELS
from src.eval import evaluate
from sys import argv
import torch
from torch_optimizer import RAdam, Lookahead
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy, one_hot
from math import inf

OUTPUT_FILE = argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")


def train(
        model,
        dataset,
        optimizer,
        batch_size=32,
        num_epochs=5,
        loss_fn=binary_cross_entropy,
        val_set=None,
        max_batches_per_epoch=inf
):
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    epoch_losses = []
    for epoch in range(num_epochs):
        print(f"\tBeginning epoch {epoch}...")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            if i >= max_batches_per_epoch:
                break
            inputs, labels = data
            inputs["tokens"] = inputs["tokens"].to(device)
            inputs["pros_feat"] = inputs["pros_feat"].to(device)
            labels = labels.to(device)
            labels = one_hot(labels, num_classes=len(POSSIBLE_LABELS)).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print cumulative batch loss every 10 mini-batches
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i+1)
                )

            # Checkpoint every 200 mini-batches
            if i % 200 == 199:
                print("Model checkpoint--saving...")
                torch.save(model, OUTPUT_FILE.replace(".pt", "-checkpoint.pt"))

        epoch_losses.append(running_loss)
        if running_loss == max(epoch_losses):
            print("Saving model...")
            torch.save(model, OUTPUT_FILE)
            print("Model saved.")

        if val_set:
            print("Validating...")
            p, r, f, p_combined, r_combined, f_combined = evaluate(model, val_set)
            print("p, r, f, p_combined, r_combined, f_combined: ", p, r, f, p_combined, r_combined, f_combined)

    print("Finished training.")


if __name__ == "__main__":
    model = AutoPuncModel()
    model.to(device)
    train_set = AutoPuncDataset("../data/train")
    dev_set = AutoPuncDataset("../data/dev")
    pretrain_set = AutoPuncDataset("../data/iwslt")

    print("Initializing optimizer...")
    radam = RAdam(model.parameters(), betas=(.9, .999), lr=1e-5, eps=1e-8)
    lookahead_optimizer = Lookahead(radam, k=6, alpha=0.5)

    print("Pretraining model...")
    train(model, pretrain_set, lookahead_optimizer, num_epochs=1, max_batches_per_epoch=1000)
    torch.save(model, OUTPUT_FILE.replace(".pt", "-pretrained.pt"))

    print("Training model...")
    train(model, train_set, lookahead_optimizer, val_set=dev_set)
