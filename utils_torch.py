import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import  pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import scipy.stats as stats

plt.rcParams['figure.constrained_layout.use'] = True

# Funzioni di utilità
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i])]
    true_name = target_names[int(y_test[i])]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4, rgb=False):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        if not rgb:
            plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        else:
            plt.imshow(images[i])
        plt.title(titles[i], size=12)
        plt.axis('off')

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 10})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=16)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    plt.show()

def print_classification_report(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\n--- Report delle metriche ---\n")
    print(report_df.to_string(float_format="{:0.2f}".format))
    return report_df

def t_test_accuracy(acc_full_batch, acc_mini_batch, acc_sgd):
    """
    Calcola il t-test per confrontare le accuracy tra i tre metodi.

    Parameters:
    acc_full_batch (list): Lista delle accuracy per il metodo Full Batch.
    acc_mini_batch (list): Lista delle accuracy per il metodo Mini Batch.
    acc_sgd (list): Lista delle accuracy per il metodo SGD.

    Returns:
    dict: Risultati del t-test sotto forma di dizionario con p-value e statistiche.
    """
    results = {}

    # Confronto Full Batch vs Mini Batch
    t_stat_fb_mb, p_value_fb_mb = ttest_ind(acc_full_batch, acc_mini_batch, equal_var=False)
    results["Full Batch vs Mini Batch"] = {
        "t_stat": t_stat_fb_mb,
        "p_value": p_value_fb_mb
    }

    # Confronto Full Batch vs SGD
    t_stat_fb_sgd, p_value_fb_sgd = ttest_ind(acc_full_batch, acc_sgd, equal_var=False)
    results["Full Batch vs SGD"] = {
        "t_stat": t_stat_fb_sgd,
        "p_value": p_value_fb_sgd
    }

    # Confronto Mini Batch vs SGD
    t_stat_mb_sgd, p_value_mb_sgd = ttest_ind(acc_mini_batch, acc_sgd, equal_var=False)
    results["Mini Batch vs SGD"] = {
        "t_stat": t_stat_mb_sgd,
        "p_value": p_value_mb_sgd
    }

    return results