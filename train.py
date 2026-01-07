import torch
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import wandb


class Trainer:
    def __init__(
        self, model, dataloaders, config, criterion, optimizer, scheduler=None
    ):
        """Simple trainer for PyTorch models"""
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_wandb = config.get("use_wandb", False)
        self.model_name = config.get("model", {}).get("architecture", "model")

    def train(self):
        """Run training loop with validation and return history"""
        best_val_loss = float("inf")
        epochs = self.config.get("training", {}).get("epochs", 10)

        # Initialize history dictionary
        history = {
            "Epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "confusion_matrices": [],
        }

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_it = tqdm(
                self.dataloaders["train"],
                desc=f"Training Epoch {epoch+1}/{epochs}",
                colour="green",
                ncols=120,
            )
            for inputs, labels in train_it:
                inputs, labels = (
                    inputs.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True).float(),
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                # Calculate training accuracy
                # outputs = torch.abs(outputs) # Get magnitude of complex outputs
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted.squeeze() == labels).sum().item()
                train_total += labels.size(0)
                train_it.set_postfix(
                    loss=f"{train_loss/len(self.dataloaders['train'].dataset):.4f}",
                    accuracy=f"{train_correct/train_total:.4f}",
                )
            print()
            train_it.close()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            val_it = tqdm(
                self.dataloaders["val"],
                desc=f"Validating Epoch {epoch+1}/{epochs}",
                colour="blue",
                ncols=120,
            )
            with torch.no_grad():
                for inputs, labels in val_it:
                    inputs, labels = (
                        inputs.to(self.device),
                        labels.to(self.device).float(),
                    )
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)

                    # Store predictions and labels for metrics
                    # outputs = torch.abs(outputs) # Get magnitude of complex outputs
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    # Safely handle predictions regardless of batch size
                    pred_np = predicted.cpu().numpy().reshape(-1)  # Flatten to 1D array
                    all_preds.extend(pred_np)
                    # Safely handle labels regardless of batch size
                    label_np = labels.cpu().numpy().reshape(-1)
                    all_labels.extend(label_np)
                    val_it.set_postfix(
                        loss=f"{val_loss/ len(self.dataloaders['val'].dataset):.4f}",
                        accuracy=f"{np.mean(np.array(all_preds) == np.array(all_labels)):.4f}",
                    )
            val_it.close()

            # Calculate epoch metrics
            train_loss = train_loss / len(self.dataloaders["train"].dataset)
            val_loss = val_loss / len(self.dataloaders["val"].dataset)
            train_acc = train_correct / train_total if train_total > 0 else 0

            # Convert to numpy arrays for sklearn metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Calculate validation metrics
            val_acc = np.mean(all_preds == all_labels) if len(all_labels) > 0 else 0
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

            # Add to history
            history["Epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["precision"].append(precision)
            history["recall"].append(recall)
            history["f1"].append(f1)
            history["confusion_matrices"].append(cm)

            # Log metrics to wandb
            if self.use_wandb:
                # Create confusion matrix figure for wandb
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.figure(figsize=(8, 6)), plt.subplot(111)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    ax=ax,
                    cmap="Blues",
                    xticklabels=["Normal", "AF"],
                    yticklabels=["Normal", "AF"],
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")

                # Log metrics and confusion matrix
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "confusion_matrix": wandb.Image(fig),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )
                plt.close(fig)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Log best model to wandb
                if self.use_wandb:
                    save_path = self.config.get("logging", {}).get(
                        "model_save_path", "./saved_models"
                    )
                    model_path = os.path.join(
                        save_path, f"{self.model_name}_model_epoch_{epoch+1}.pth"
                    )
                    # wandb.save(model_path) # to save models online
            self.save_model(epoch)
            print(f"✓ Model saved (val_loss: {val_loss:.4f})")

            # Step the scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Current learning rate: {current_lr}")
                wandb.log({"learning_rate": current_lr})

            print()

        # Return the training history
        return history

    def save_model(self, epoch=None):
        """Save model to disk"""
        save_path = self.config.get("logging", {}).get(
            "model_save_path", "./saved_models"
        )
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(
            save_path, f"{self.model_name}_model_epoch_{epoch+1}.pth"
        )
        torch.save(self.model.module.state_dict(), model_path)
