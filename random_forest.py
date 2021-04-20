from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from helpers.scorers import scorers_dict, generate_report
from helpers.data import get_data


if __name__ == "__main__":
    X, y = get_data()


    RF_classifier = RandomForestClassifier()

    model = make_pipeline(TfidfVectorizer(sublinear_tf=True, use_idf=True, ngram_range=(1, 2)), RF_classifier)

    # Evaluate classifier using KFold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=77)
    scores = cross_validate(model, X, y, scoring=scorers_dict, cv=kf, n_jobs=-1)
    generate_report(scores)