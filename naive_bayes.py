from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_validate
from helpers.scorers import scorers_dict, generate_report
from helpers.data import get_data, get_data_with_features, ExtractColumn


def naive_bayes_classifier(use_stories=False, use_custom_features=False, use_class_grouping=False):
    if use_custom_features:
        # Use TF-IDF together with custom features
        X, y = get_data_with_features(include_stories=use_stories, use_class_grouping=use_class_grouping)

        tfidf_pipe = make_pipeline(ExtractColumn(0), TfidfVectorizer(sublinear_tf=True, use_idf=True))
        custom_pipe = make_pipeline(ExtractColumn(1))

        # Naive Bayes classifier pipeline
        model = make_pipeline(FeatureUnion([("tfidf", tfidf_pipe), ("custom_features", custom_pipe)]),
                              MultinomialNB(alpha=0.01))
    else:
        # TF-IDF only
        X, y = get_data(include_stories=use_stories, use_class_grouping=use_class_grouping)

        # Naive Bayes classifier pipeline
        model = make_pipeline(TfidfVectorizer(sublinear_tf=True, use_idf=True, ngram_range=(1, 2)),
                              MultinomialNB(alpha=0.01))

    # Evaluate classifier using KFold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=77)
    scores = cross_validate(model, X, y, scoring=scorers_dict, cv=kf, n_jobs=-1)
    generate_report(scores)
