import torch
import os
import PIL
import torch
import mlflow
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torchvision import datasets
from torchmetrics.classification import MulticlassROC, MulticlassAUROC

RANDOM_SEED = 23

def create_dataloaders(
    train_path: Path | str,
    valid_path: Path | str,
    test_path: Path | str,
    train_transforms: torch.nn.Module,
    valid_transforms: torch.nn.Module,
    test_transforms: torch.nn.Module,
    batch_size: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns dataloaders 
    for train, valid and test dataset
    """
    
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms, target_transform=None)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return (train_dataloader, valid_dataloader, test_dataloader)

def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    epochs: int
) -> Dict[str, List[float]]:
    """
    Trains model and returns metrics
    """

    current_directory_path = Path.cwd()

    with mlflow.start_run():
        result = {
            "train_acc": [],
            "train_loss": [],
            "valid_acc": [],
            "valid_loss": []
        }

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            valid_loss, valid_acc = test_step(
                model=model,
                dataloader=valid_dataloader,
                criterion=criterion,
                device=device
            )

            if (epoch + 1) % 10 == 0:
                save_model(model=model, directory=current_directory_path/"models", epoch=epoch+1)

            print(f"Epoch = {epoch+1} | {train_loss=:.3f} | {train_acc=:.3f} | {valid_loss=:.3f} | {valid_acc=:.3f}")

            result["train_acc"].append(train_acc)
            result["train_loss"].append(train_loss)
            result["valid_acc"].append(valid_acc)
            result["valid_loss"].append(valid_loss)

            mlflow.log_metrics(
                metrics={
                    "Train accuracy": train_acc,
                    "Validation accuracy": valid_acc,
                    "Train loss": train_loss,
                    "Validation loss": valid_loss
                },
                step=epoch+1
            )
        
        mlflow.pytorch.log_model(pytorch_model=model, name="model")
        mlflow.set_tags({
            "model type": "cnn",
            "dataset": "RSL dataset",
            "framework": "pytorch"
        })

        return result

def accuracy(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> float:
    """
    Calculates accuracy (amount of prediction and target matches)
    """
    return (pred == target).sum().item() / len(pred)

def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Performs full train epoch
    """
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device).to(memory_format=torch.channels_last), targets.to(device)

        y_pred_logits = model(features)
        loss = criterion(y_pred_logits, targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_labels = y_pred_logits.argmax(dim=1)
        train_acc += accuracy(pred=y_pred_labels, target=targets)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return (train_loss, train_acc)

def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Performs full validation/test epoch
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (features, targets) in enumerate(dataloader):
            features, targets = features.to(device).to(memory_format=torch.channels_last), targets.to(device)

            y_pred_logits = model(features)
            loss = criterion(y_pred_logits, targets)
            test_loss += loss.item()

            y_pred_labels = y_pred_logits.argmax(dim=1)
            test_acc += accuracy(pred=y_pred_labels, target=targets)
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return (test_loss, test_acc)

def save_model(
    model: torch.nn.Module,
    directory: Path | str,
    filename: str = None,
    epoch: int | str = None
) -> None:
    """
    Saves model weights into specific directory
    """
    if not filename and not epoch:
        filename = f"rslr_model.pth"
    elif not filename:
        filename = f"rslr_model_epoch_{epoch}.pth"
    
    save_path = os.path.join(directory, filename)
    torch.save(obj=model.state_dict(), f=save_path)

def calc_and_plot_roc_auc(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> float:
    """
    Run model (preferably on test dataloader), plots ROC and calcs AUC
    """
    # Evaluate model
    model.eval()
    with torch.inference_mode():
        # Setup ROC and AUC
        classes = dataloader.dataset.classes
        metric_roc = MulticlassROC(num_classes=len(classes), average=None) # ROC per-class
        metric_auroc = MulticlassAUROC(num_classes=len(classes), average="macro") # macro-averaged AUC

        for batch, (features, target) in enumerate(dataloader):
            features, target = features.to(device).to(memory_format=torch.channels_last), target.to(device)

            pred_logits = model(features)
            metric_roc.update(pred_logits, target)
            metric_auroc.update(pred_logits, target)

        macro_auroc = metric_auroc.compute()
        fig, ax = metric_roc.plot(score=True)
        fig.set_size_inches(9, 9)
        plt.title(f"Multiclass ROC | Macro-averaged AUROC score: {macro_auroc:.3f}")
        
        # Clarify legend classes
        _, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for i, label in enumerate(labels):
            score_part = label.split(' ')[1]
            new_labels.append(f"Буква \'{classes[i]}\' {score_part}")
        plt.legend(new_labels)
        plt.show()

        return macro_auroc