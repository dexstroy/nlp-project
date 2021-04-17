from sklearn import preprocessing
from helpers.constants import ALL_CLASSES, DATASET_PATH
from helpers.worksheet import read_worksheet
from helpers.preprocess import preprocess_data


def get_data():
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)  # Encode classes with numerical labels
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11)
    X.extend(X2)  # Join worksheets into a single dataset
    y.extend(y2)
    return preprocess_data(X, y)
