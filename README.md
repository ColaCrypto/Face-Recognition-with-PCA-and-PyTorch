# Face Recognition with Principal Component Analysis (PCA)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-blue.svg)
![NumPy](https://img.shields.io/badge/Numpy-1.24-green.svg)

## Project Overview

This project provides a comprehensive, from-scratch implementation of the classic **Eigenfaces** method for face recognition, combining Principal Component Analysis (PCA) for dimensionality reduction with a PyTorch-based classifier.

The primary goal is to demonstrate a deep understanding of the entire machine learning pipeline: from data exploration and preprocessing to feature extraction, model training, and rigorous comparative analysis. The project systematically evaluates the impact of PCA on performance and compares three distinct gradient descent training strategies (Full-Batch, Mini-Batch, and SGD), validating the results with statistical tests (T-Test).

This work was developed for the "Statistical and Mathematical Methods For AI" course at the Polytechnic University of Bari.

---

## Key Concepts and Features Implemented

*   **Dimensionality Reduction with PCA**: A from-scratch-like, modular implementation of the Eigenfaces technique to transform high-dimensional image data (32,256 features) into a compact, low-dimensional feature space (70 components), capturing over 95% of the dataset's variance.
*   **PyTorch-based Classifier**: A custom Logistic Regression model built in PyTorch to perform multi-class classification on the extracted PCA features.
*   **Comparative Training Analysis**: A systematic evaluation of three gradient descent optimization strategies to analyze the trade-offs between convergence speed, stability, and final accuracy:
    1.  **Full-Batch Gradient Descent**
    2.  **Mini-Batch Gradient Descent**
    3.  **Stochastic Gradient Descent (SGD)**
*   **Statistical Validation**: Employment of statistical **T-Tests** to rigorously confirm that the observed differences in accuracy between the training methods are statistically significant.
*   **Modular and Reproducible Code**: The project is structured into logical, reusable modules for data loading, PCA computation, model training, and utilities, ensuring clarity and ease of reproduction.

---

## Project Structure

The codebase is organized in a modular fashion to ensure clarity and separation of concerns.

-   `allFaces.mat`: The raw dataset from the "Extended Yale Face Database B". **(Note: This file is not included in the repository and must be downloaded separately)**.
-   `main.py`: The main entry point to run the complete PCA-based workflow, including data processing, training of all three optimizer variants, and final evaluation.
-   `no_pca.py`: A control script to run the same training process on the raw, high-dimensional data to demonstrate the necessity and effectiveness of PCA.
-   `dataset.py`: A script for initial data exploration, visualization, and understanding of the dataset's structure and distribution.
-   `modular_pca.py`: Contains all functions related to PCA, including data loading, splitting, PCA computation, and visualization of eigenfaces and reconstructions.
-   `modular_training.py`: Encapsulates the PyTorch training and evaluation loops, abstracting the logic for reuse across different experiments.
-   `torch_model.py`: Defines the `LogisticRegressionModel` class, the core PyTorch classification model.
-   `utils_torch.py`: A utility module containing helper functions for plotting results, confusion matrices, classification reports, and performing T-Tests.

---

## Getting Started

### Prerequisites
*   Python 3.9+
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TuoNomeUtente/nome-repository.git
    cd nome-repository
    ```

2.  **Download the Dataset:**
    The `allFaces.mat` dataset is not included in this repository due to its size. Please download it from (https://vision.ucsd.edu/datasets/extended-yale-face-database-b-b) and place it in the project's root directory.

3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

The project includes two main executable scripts:

1.  **To run the main experiment with PCA:**
    This script will perform the entire workflow: load data, apply PCA, train the model with Full-Batch, Mini-Batch, and SGD, and generate all comparison plots and reports.
    ```bash
    python main.py
    ```

2.  **To run the control experiment without PCA:**
    This script demonstrates the performance bottleneck and poor results when training on the raw high-dimensional data.
    ```bash
    python no_pca.py
    ```

---

## Project Presentation

For a detailed walkthrough of the project's objectives, challenges, methodology, and results, you can view the complete PowerPoint presentation.

**https://github.com/ColaCrypto/Face-Recognition-with-PCA-and-PyTorch/blob/main/PPT_Face-Recognition-with-PCA-and-PyTorch.pdf**

