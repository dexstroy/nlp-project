import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from helpers.worksheet import read_worksheet
from helpers.preprocess import preprocess_data
from helpers.constants import ALL_CLASSES, DATASET_PATH

if __name__ == "__main__":
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)  # Encode classes with numerical labels
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11)
    X.extend(X2)  # Join worksheets into a single dataset
    y.extend(y2)
    X, y = preprocess_data(X, y)

    # Naive Bayes classifier pipeline
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True, use_idf=True, ngram_range=(1, 1)), MultinomialNB(alpha=.01))

    # Evaluate classifier using KFold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=77)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=kf, n_jobs=-1)
    print("Accuracy: {} {}".format(np.mean(scores), np.std(scores)))

    # Evaluate classifier with a single train/test dataset split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=77)
    # model.fit(X_train, y_train)
    # predicted_cats = model.predict(X_test)
    # print(predicted_cats)
    # print("The accuracy is {}".format(accuracy_score(y_test, predicted_cats)))
    # print(confusion_matrix(y_test, predicted_cats))
