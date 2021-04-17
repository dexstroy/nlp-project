import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate
from helpers.scorers import scorers_dict, generate_report
from helpers.data import get_data


if __name__ == "__main__":
    X, y = get_data()

    # Naive Bayes classifier pipeline
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True, use_idf=True, ngram_range=(1, 2)), MultinomialNB(alpha=0.01))

    # Evaluate classifier using KFold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=77)
    scores = cross_validate(model, X, y, scoring=scorers_dict, cv=kf, n_jobs=-1)
    generate_report(scores)

    # Evaluate classifier with a single train/test dataset split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=77)
    # model.fit(X_train, y_train)
    # predicted_cats = model.predict(X_test)
    # print(predicted_cats)
    # print("The accuracy is {}".format(accuracy_score(y_test, predicted_cats)))
    # print(confusion_matrix(y_test, predicted_cats))
