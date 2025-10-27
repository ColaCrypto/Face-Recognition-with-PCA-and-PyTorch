import matplotlib.pyplot as plt

from modular_pca import setup_plot_style, load_data, split_dataset
from modular_training import prepare_dataloader, train_model, plot_results
from torch_model import LogisticRegressionModel
from utils_torch import (title, plot_gallery, print_classification_report, plot_confusion_matrix, t_test_accuracy)
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Workflow senza PCA
setup_plot_style()

# Parametri
file_path = 'allFaces.mat'
test_size = 0.4

# Caricamento dati
faces, m, n, nfaces = load_data(file_path)

# Suddivisione del dataset
X_train, X_test, y_train, y_test = split_dataset(faces, nfaces, test_size)

# Prepara i tensori
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Modello Logistic Regression
input_dim = X_train.shape[1]
num_classes = len(set(y_train))
model = LogisticRegressionModel(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()


# Full-Batch Training
optimizer_batch = optim.SGD(model.parameters(), lr=1)
train_loader_full, test_loader_full = prepare_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=X_train.shape[0])
train_losses_full, train_accuracies_full, test_losses_full, test_accuracies_full = train_model(
    model, train_loader_full, test_loader_full, criterion, optimizer_batch, num_epochs=150
)
plot_results(train_losses_full, train_accuracies_full, test_losses_full, test_accuracies_full, title_suffix="(Full-Batch)")

# Mini-Batch Training
optimizer_mini = optim.SGD(model.parameters(), lr=0.01)
train_loader_mini, test_loader_mini = prepare_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=32)
train_losses_mini, train_accuracies_mini, test_losses_mini, test_accuracies_mini = train_model(
    model, train_loader_mini, test_loader_mini, criterion, optimizer_mini, num_epochs=150
)
plot_results(train_losses_mini, train_accuracies_mini, test_losses_mini, test_accuracies_mini, title_suffix="(Mini Batch)")

# SGD Training
optimizer_sgd = optim.SGD(model.parameters(), lr=0.001)
train_loader_sgd, test_loader_sgd = prepare_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=1)
train_losses_sgd, train_accuracies_sgd, test_losses_sgd, test_accuracies_sgd = train_model(
    model, train_loader_sgd, test_loader_sgd, criterion, optimizer_sgd, num_epochs=150
)
plot_results(train_losses_sgd, train_accuracies_sgd, test_losses_sgd, test_accuracies_sgd, title_suffix="(SGD)")

# valutazione
target_names = [f"Person {i+1}" for i in range(len(np.unique(y_test)))]

print("\n--- Risultati Full Batch ---")
report_batch = print_classification_report(y_test, torch.argmax(model(X_test_tensor), 1), target_names)
plot_confusion_matrix(y_test, torch.argmax(model(X_test_tensor), 1), target_names, title='Matrice di Confusione - Full-Batch')

print("\n--- Risultati Mini Batch ---")
report_mini = print_classification_report(y_test, torch.argmax(model(X_test_tensor), 1), target_names)
plot_confusion_matrix(y_test, torch.argmax(model(X_test_tensor), 1), target_names, title='Matrice di Confusione - Mini Batch')

print("\n--- Risultati SGD ---")
report_sgd = print_classification_report(y_test, torch.argmax(model(X_test_tensor), 1), target_names)
plot_confusion_matrix(y_test, torch.argmax(model(X_test_tensor), 1), target_names, title='Matrice di Confusione - SGD')

# Confronto finale delle metriche
print("\n--- Confronto delle metriche tra i metodi ---")
metrics_comparison = {
    "Metodo": ["Full Batch", "Mini Batch", "SGD"],
    "Accuracy": [
        report_batch.loc["accuracy"].values[0],
        report_mini.loc["accuracy"].values[0],
        report_sgd.loc["accuracy"].values[0],
    ],
    "Precision (avg)": [
        report_batch.loc["macro avg", "precision"],
        report_mini.loc["macro avg", "precision"],
        report_sgd.loc["macro avg", "precision"],
    ],
    "Recall (avg)": [
        report_batch.loc["macro avg", "recall"],
        report_mini.loc["macro avg", "recall"],
        report_sgd.loc["macro avg", "recall"],
    ],
    "F1-Score (avg)": [
        report_batch.loc["macro avg", "f1-score"],
        report_mini.loc["macro avg", "f1-score"],
        report_sgd.loc["macro avg", "f1-score"],
    ]
}

comparison_df = pd.DataFrame(metrics_comparison)
print(comparison_df.to_string(index=False, float_format="{:0.2f}".format))

# Calcolo del t-test
t_test_results = t_test_accuracy(test_accuracies_full, test_accuracies_mini, test_accuracies_sgd)

# Stampa dei risultati
for comparison, stats in t_test_results.items():
    print(f"{comparison}: t_stat={stats['t_stat']:.4f}, p_value={stats['p_value']:.100f}")