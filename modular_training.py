import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time


def prepare_dataloader(X_train, y_train, X_test, y_test, batch_size):
    """Prepara i DataLoader per il training e il testing."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer):
    """Esegue un'epoca di training."""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss /= total
    accuracy = correct / total
    return epoch_loss, accuracy


def evaluate_model(model, test_loader, criterion):
    """Valuta il modello sul test set."""
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    epoch_loss /= total
    accuracy = correct / total
    return epoch_loss, accuracy


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    """Addestra il modello e restituisce i risultati del training."""
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in: {total_time:.2f} seconds \n")

    return train_losses, train_accuracies, test_losses, test_accuracies


def plot_results(train_losses, train_accuracies, test_losses, test_accuracies, title_suffix=""):
    """Visualizza i risultati del training e del test."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch {title_suffix}')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch {title_suffix}')
    plt.legend()

    plt.show()