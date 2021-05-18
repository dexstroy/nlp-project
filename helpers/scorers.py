from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
import numpy as np


def accuracy_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision_scorer_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=1, average="macro")


def precision_scorer_weighted(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=1, average="weighted")


def recall_scorer_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=1, average="macro")


def recall_scorer_weighted(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=1, average="weighted")


def f1_scorer_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=1, average="macro")


def f1_scorer_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=1, average="weighted")


def generate_report(scores):
    print("{:^8} {:^8} {:^8} {:^8}".format("", "Precison", "Recall", "F1 Score"))
    print("{:^8} {:^8.3f} {:^8.3f} {:^8.3f}".format("Macro", np.mean(scores["test_precision_macro"]), np.mean(scores["test_recall_macro"]), np.mean(scores["test_f1_macro"])))
    print("{:^8} {:^8.3f} {:^8.3f} {:^8.3f}".format("Weighted", np.mean(scores["test_precision_weighted"]), np.mean(scores["test_recall_weighted"]), np.mean(scores["test_f1_weighted"])))
    print("------------------------------------------------")
    print("{:^8} {:^8.3f}".format("Accuracy", np.mean(scores["test_accuracy"])))


scorers_dict = {
    "accuracy": make_scorer(accuracy_scorer),
    "f1_weighted": make_scorer(f1_scorer_weighted),
    "f1_macro": make_scorer(f1_scorer_macro),
    "recall_weighted": make_scorer(recall_scorer_weighted),
    "recall_macro": make_scorer(recall_scorer_macro),
    "precision_weighted": make_scorer(recall_scorer_weighted),
    "precision_macro": make_scorer(recall_scorer_macro),
}
