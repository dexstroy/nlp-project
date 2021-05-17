from sklearn import preprocessing
from helpers.constants import ALL_CLASSES, DATASET_PATH, STORIES_PATH
from helpers.worksheet import read_worksheet
from helpers.stories import read_stories
from helpers.preprocess import preprocess_data
from helpers.custom_features import get_custom_features


def get_data(include_stories=False):
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)  # Encode classes with numerical labels
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11)
    X.extend(X2)  # Join worksheets into a single dataset
    y.extend(y2)
    if include_stories:
        # include stories as content discussion
        X_s, y_s = read_stories(STORIES_PATH, le)
        X.extend(X_s)
        y.extend(y_s)
    return preprocess_data(X, y)


def get_data_with_features(include_stories=False):
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)  # Encode classes with numerical labels
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11)
    X.extend(X2)  # Join worksheets into a single dataset
    y.extend(y2)
    if include_stories:
        # include stories as content discussion
        X_s, y_s = read_stories(STORIES_PATH, le)
        X.extend(X_s)
        y.extend(y_s)
    return get_custom_features(X, y)
