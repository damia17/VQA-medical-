import os
import warnings
import torch
from torch import nn
from torch.nn.parallel import DataParallel
warnings.filterwarnings("ignore")
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from MED_VQA_Data_Word2Vec import MED_VQA_Data
from model import VQAClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]+1
        results_df = checkpoint["results_df"]
        print(f"Checkpoint loaded. Resuming training from epoch {epoch + 1}")
        return epoch, results_df
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, pd.DataFrame(
            columns=[
                "Epoch",
                "Training loss",
                "Training accuracy",
                "Validation accuracy",
                "Test accuracy",
                "Precision",
                "Recall",
                "F1",
            ]
        )

# Hyperparameters
input_size = 768
num_classes = 1548 
learning_rate = 0.0001
num_epochs = 30
batch_size = 128
dropout_prob = 0.2
num_heads = 4

# # Load dataset
train_df = pd.read_csv("../datasets/clef2019/train/traindf_labeled.csv")
val_df = pd.read_csv("../datasets/clef2019/valid/valdf_labeled.csv")
test_df = pd.read_csv("../datasets/clef2019/test/testdf_labeled.csv")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Create DataLoader
train_dataset = MED_VQA_Data(train_df)
val_dataset = MED_VQA_Data(val_df)
test_dataset = MED_VQA_Data(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = VQAClassifier(
    embed_dim=input_size, num_classes=num_classes, dropout=dropout_prob, num_heads=num_heads
)

model = DataParallel(model)

print(f"model's size : {count_parameters(model)}")

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Function to calculate metrics
def calculate_metrics(loader, model):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data in loader:
            image_emb = data["img_emb"].to(device)
            text_emb = data["text_emb"].to(device)
            labels = data["label"].to(device)
            type = data["type"]
            outputs = model(image_emb, text_emb)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / total
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    return accuracy, avg_loss, precision, recall, f1


# Training loop
results_df = pd.DataFrame(
    columns=[
        "Epoch",
        "Training loss",
        "Validation accuracy",
        "Test accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
)


# Load the latest checkpoint
checkpoint_prefix = f"vqa_model_{num_heads}h_{dropout_prob}dropout_wv"
latest_checkpoint_path = f"checkpoints/{checkpoint_prefix}_latest.pth"
start_epoch, results_df = load_checkpoint(model, optimizer, latest_checkpoint_path)

# Training loop with checkpoints
checkpoint_interval = 1  # Save checkpoint every 5 epochs
checkpoint_prefix = "vqa_model_{num_heads}h_{dropout_prob}dropout_wv"


for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
    )
    batch_losses = []
    for batch_idx, data in enumerate(train_loader_tqdm):
        image_emb = data["img_emb"].to(device)
        text_emb = data["text_emb"].to(device)
        labels = data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(image_emb, text_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = sum(batch_losses) / len(batch_losses)
    val_acc, _, _, _, _, _ = calculate_metrics(val_loader, model)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f},  Val Acc: {val_acc:.4f}"
    )

    test_acc, _, pre_score, rec_score, f1, _ = calculate_metrics(
        test_loader, model
    )
    print(
        f'Test Acc: {test_acc:.4f}, Precision: {pre_score:.4f}, Recall: {rec_score:.4f}, F1: {f1:.4f}'
    )
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    [
                        epoch,
                        train_loss,
                        val_acc,
                        test_acc,
                        pre_score,
                        rec_score,
                        f1,
                    ]
                ],
                columns=[
                    "Epoch",
                    "Training loss",
                    "Validation accuracy",
                    "Test accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                ],
            ),
        ]
    )
    results_df.to_csv(f"results_{num_heads}h_{dropout_prob}dropout_wv.csv", index=False)

    # Save checkpoint
    
    checkpoint_path = f"checkpoints/{checkpoint_prefix}_{epoch+1}.pth"
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results_df": results_df,
            },
            checkpoint_path,
        )

    checkpoint_path = f"checkpoints/{checkpoint_prefix}_latest.pth"
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results_df": results_df,
            },
            checkpoint_path,
        )

    print(f"Checkpoint saved at epoch {epoch+1}")


# Save final training results
print("Training complete.")
final_checkpoint_path = f"checkpoints/{checkpoint_prefix}_final.pth"
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "results_df": results_df,
    },
    final_checkpoint_path,
)
print(f"Final checkpoint saved at epoch {epoch}")

