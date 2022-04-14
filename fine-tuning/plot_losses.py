import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("history_file", help="path to history file")
parser.add_argument("--save_to", "-s", help="path to save figure")
args = parser.parse_args()

epochs, train_losses, val_losses, val_accs = [], [], [], []
with open(args.history_file, "r") as fhist:
    for line in fhist:
        epoch, train_loss, val_loss, val_acc = line.strip().split('\t')
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))

plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label="train")
plt.plot(epochs, val_losses, label="va;")
plt.legend(loc="best")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.subplot(2, 1, 2)
plt.plot(epochs, val_accs, label="val")
plt.legend(loc="best")
plt.xlabel("epochs")
plt.ylabel("accuracy")

if args.save_to is None:
    _ = plt.show()
else:
    plt.savefig(args.save_to)