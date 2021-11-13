import os
import pickle

from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             roc_curve,
                             roc_auc_score
                            )


def evaluate_model(y_test, y_pred, model_name):
    
    print(f"Evaluating {model_name}...\n")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    obj = {
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": cm, # [[TN,FP],[FN,TP]]
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "tnr": cm[0][0]/(cm[0][0]+cm[0][1]),
        "roc_auc_score": roc_auc_score(y_test, y_pred)
    }

    return obj


def save_pickle_model(model, model_path):

    pickle.dump(model, open(model_path, 'wb'))


def load_pickle_model(filepath):

    return pickle.load(open(filepath, 'rb'))
