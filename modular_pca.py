import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io
import time

# Impostazioni grafiche
def setup_plot_style():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams.update({'font.size': 17})
    plt.rcParams['figure.constrained_layout.use'] = True

# Caricamento dei dati
def load_data(file_path):
    mat_contents = scipy.io.loadmat(file_path)
    faces = mat_contents['faces']
    m, n = int(mat_contents['m']), int(mat_contents['n'])
    nfaces = np.ndarray.flatten(mat_contents['nfaces'])
    return faces, m, n, nfaces

# Suddivisione del dataset
def split_dataset(faces, nfaces, test_size=0.4):
    X = faces.T
    y = np.concatenate([[i] * n for i, n in enumerate(nfaces)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test

# Calcolo della PCA
def compute_pca(X_train, n_components):
    print(f"Calcolo della PCA con {n_components} componenti...")
    t0 = time.time()
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    print(f"Completato in {time.time() - t0:.3f}s")
    return pca

# Visualizzazione dello scree plot
def plot_scree(explained_variance_ratio, n_components):
    indices = np.arange(0, len(explained_variance_ratio), 5)
    cum_explained_variance_ = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 7))
    plt.plot(indices + 1, explained_variance_ratio[indices], marker='o', linestyle='--', color='b',
             label='Explained Variance')
    plt.plot(indices + 1, cum_explained_variance_[indices], marker='o', linestyle='-', color='r',
             label='Cumulative Explained Variance')
    plt.xticks(np.arange(0, n_components, 10))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('Number of components')
    plt.ylabel('Explained Variance')
    plt.title('Scree Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualizzazione delle prime eigenfaces
def plot_eigenfaces(eigenfaces, m, n):
    fig1, axes = plt.subplots(1, 5, figsize=(15, 5), constrained_layout=True)
    for i in range(5):
        ax = axes[i]
        img = ax.imshow(eigenfaces[i].reshape((m, n)).T, cmap='gray')
        ax.set_title(f"Eigenface {i+1}")
        ax.axis('off')
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

# Plot delle immagini ricostruite
def plot_reconstructed_images(testFace, avgFace, pca_components, r_values, m, n):
    testFaceMS = testFace - avgFace
    height = int(np.ceil((len(r_values) + 1) / 4))  # Include l'immagine originale
    fig, axes = plt.subplots(height, 4, figsize=(12, 8))
    axes = axes.ravel()  # Appiattisce l'array di assi per facilitare l'iterazione

    # Plotta l'immagine originale
    axes[0].imshow(np.reshape(testFace, (m, n)).T, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Calcolo e plot delle immagini ricostruite
    for i, r in enumerate(r_values):
        reconFace = avgFace + pca_components[:r].T @ (pca_components[:r] @ testFaceMS)
        axes[i+1].imshow(np.reshape(reconFace, (m, n)).T, cmap='gray')
        axes[i+1].set_title(f'eigenfaces = {r}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

# Salvataggio caratteristiche PCA
def save_pca_features(pca, faces, nfaces, person, num_images):
    subset = faces[:, sum(nfaces[:person]):sum(nfaces[:(person + 1)])]
    subset = subset[:, :num_images]
    data = pca.transform(subset.T)
    return data