import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, roc_auc_score, roc_curve
)

from models.models import GraphModel, BaselineModel


PLOTS_PATH = Path("plots")



def encoder_decoder_loss(x_in: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_in, x_hat)

def disease_prediction_loss(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(probs, labels, reduction='mean')


class TrainGraphEncoder(nn.Module):
    def __init__(self, architect_params: dict, lr: float, num_epochs: int, batch_size: int, data):
        super().__init__()
        self.architect_params = architect_params
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = GraphModel(architect_params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.mask = architect_params['mask']
        self.x_train, self.y_train, self.x_val, self.y_val = data
        # Create all parent directories if they don't exist
        PLOTS_PATH.parent.mkdir(parents=True, exist_ok=True)


    def gradient_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_emb, x_hat, p = self.model(x, self.mask)
        ed_loss = encoder_decoder_loss(x, x_hat)
        disease_loss = disease_prediction_loss(p, y)
        loss = 0.5*ed_loss + 0.5*disease_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_model(self) -> float:
        self.model.eval()
        x_emb, x_hat, p = self.model(self.x_val, self.mask)
        ed_loss = encoder_decoder_loss(self.x_val, x_hat)
        disease_loss = disease_prediction_loss(p, self.y_val)
        loss = 0.5 * ed_loss + 0.5 * disease_loss
        self.model.train()
        return loss.item()

    def run_epochs(self):
        train_losses = []
        val_losses = []
        for e in tqdm(range(self.num_epochs), desc=f"training graph-aware model"):
            batch_idxs = random.sample(range(self.x_train.shape[0]), self.batch_size)
            x_b, y_b = self.x_train[batch_idxs], self.y_train[batch_idxs]
            train_loss = self.gradient_step(x_b, y_b)
            train_losses.append(train_loss)
            if e % 100 == 0:
                val_losses.append(self.eval_model())

        smth_train_losses = torch.tensor(train_losses[:(len(train_losses) - len(train_losses) % 100)]).view(-1, 100).mean(dim=1)
        padded_dim = min(len(smth_train_losses), len(val_losses))
        plt.plot(range(padded_dim), smth_train_losses[:padded_dim], label="training loss")
        plt.plot(range(padded_dim), val_losses[:padded_dim], label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(Path(PLOTS_PATH / 'loss_curve_graph.png'))


class TrainBaselineEncoder(nn.Module):
    def __init__(self, architect_params: dict, lr: float, num_epochs: int, batch_size: int, data):
        super().__init__()
        self.architect_params = architect_params
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = BaselineModel(architect_params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.x_train, self.y_train, self.x_val, self.y_val = data

    def gradient_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_emb, x_hat, p = self.model(x)
        ed_loss = encoder_decoder_loss(x, x_hat)
        disease_loss = disease_prediction_loss(p, y)
        loss = 0.5 * ed_loss + 0.5 * disease_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_model(self) -> float:
        self.model.eval()
        x_emb, x_hat, p = self.model(self.x_val)
        ed_loss = encoder_decoder_loss(self.x_val, x_hat)
        disease_loss = disease_prediction_loss(p, self.y_val)
        loss = 0.5 * ed_loss + 0.5 * disease_loss
        self.model.train()
        return loss.item()

    def run_epochs(self):
        train_losses = []
        val_losses = []
        for e in tqdm(range(self.num_epochs), desc=f"training baseline model"):
            batch_idxs = random.sample(range(self.x_train.shape[0]), self.batch_size)
            x_b, y_b = self.x_train[batch_idxs], self.y_train[batch_idxs]
            train_loss = self.gradient_step(x_b, y_b)
            train_losses.append(train_loss)
            if e % 100 == 0:
                val_losses.append(self.eval_model())

        smth_train_losses = torch.tensor(train_losses[:(len(train_losses) - len(train_losses) % 100)]).view(-1, 100).mean(dim=1)
        padded_dim = min(len(smth_train_losses), len(val_losses))
        plt.plot(range(padded_dim), smth_train_losses[:padded_dim], label="training loss")
        plt.plot(range(padded_dim), val_losses[:padded_dim], label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(Path(PLOTS_PATH / 'loss_curve_baseline.png'))


@torch.no_grad()
def compare_models(graph_model: GraphModel, baseline_model: BaselineModel, data, mask) -> None:
    x, y = data
    _, _, p_graph = graph_model(x, mask)
    _, _, p_base = baseline_model(x)
    pred_graph = (p_graph > 0.5).to(dtype=torch.long)
    pred_base = (p_graph > 0.5).to(dtype=torch.long)

    # Confusion matrices
    conf_mat_graph = confusion_matrix(y, pred_graph)
    conf_mat_base = confusion_matrix(y, pred_base)
    disp_cm_graph = ConfusionMatrixDisplay(conf_mat_graph)
    disp_cm_base = ConfusionMatrixDisplay(conf_mat_base)

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    # Plot each confusion matrix on its respective axis
    disp_cm_graph.plot(ax=axes[0])
    axes[0].set_title('Graph Model Confusion Matrix')

    disp_cm_base.plot(ax=axes[1])
    axes[1].set_title('Baseline Model Confusion Matrix')

    plt.tight_layout()
    plt.savefig(Path(PLOTS_PATH / 'confusion_matrices.png'))

    # Accuracy and F1 scores
    acc_graph = accuracy_score(y, pred_graph)
    acc_base = accuracy_score(y, pred_base)

    f1_graph = f1_score(y, pred_graph)
    f1_base = f1_score(y, pred_base)

    # Print summary
    print("=== Accuracy Model Performance ===")
    print(f"\nGraph Model:")
    print(f"  Accuracy: {acc_graph:.4f}")
    print(f"  F1 Score: {f1_graph:.4f}")

    print(f"\nBaseline Model:")
    print(f"  Accuracy: {acc_base:.4f}")
    print(f"  F1 Score: {f1_base:.4f}")

    # ROC curves and AUC:
    fpr_graph, tpr_graph, _ = roc_curve(y, p_graph)
    fpr_base, tpr_base, _ = roc_curve(y, p_base)

    auc_graph = roc_auc_score(y, p_graph)
    auc_base = roc_auc_score(y, p_base)

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_graph, tpr_graph, label=f'Graph Model (AUC = {auc_graph:.3f})', linewidth=2)
    plt.plot(fpr_base, tpr_base, label=f'Baseline Model (AUC = {auc_base:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(Path(PLOTS_PATH / 'roc_curve.png'))

    print(f"Graph Model AUC: {auc_graph:.4f}")
    print(f"Baseline Model AUC: {auc_base:.4f}")



