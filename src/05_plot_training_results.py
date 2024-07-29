from plots import *
from helpers import *

ensure_directory("../plots")

log_file = "../results/training_log.jsonl"
data = load_log_data(log_file)

json_file_path = "../data/kuzushiji-arrays/index.json"
id_to_char = load_id_to_char_map(json_file_path)

setup_plot_style()

# Plot training and validation metrics
plot_metrics(data, ["train_loss", "val_loss"], "Training and Validation Loss", "../plots/loss_curve.png")
plot_metrics(data, ["train_accuracy", "val_accuracy"],
             "Training and Validation Accuracy", "../plots/accuracy_curve.png")
plot_metrics(data, ["train_precision", "val_precision", "train_recall", "val_recall"],
             "Precision and Recall", "../plots/precision_recall_curve.png")

# Plot top misclassifications for the last epoch
plot_top_misclassifications(data, len(data), id_to_char, top_n=50, filename="../plots/top_misclassifications_final.png")

# Plot learning curves
epochs = list(range(1, len(data) + 1))
train_loss = [entry["train_loss"] for entry in data]
val_loss = [entry["val_loss"] for entry in data]
train_acc = [entry["train_accuracy"] for entry in data]
val_acc = [entry["val_accuracy"] for entry in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1.plot(epochs, train_loss, label="Train Loss", color=colors[0])
ax1.plot(epochs, val_loss, label="Validation Loss", color=colors[1])
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(loc="upper right")
ax1.set_title("Learning Curves", fontsize=16, fontweight="bold")

ax2.plot(epochs, train_acc, label="Train Accuracy", color=colors[2])
ax2.plot(epochs, val_acc, label="Validation Accuracy", color=colors[3])
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig("../plots/learning_curves.png", dpi=500, bbox_inches="tight")
plt.close()
