from sklearn import preprocessing
from helpers.worksheet import read_worksheet
from helpers.constants import ALL_CLASSES, DATASET_PATH

if __name__ == "__main__":
    # initialize label encoder
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)

    # load worksheets - include all columns
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10, only_messages=False)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11, only_messages=False)
    
    # join worksheets into a single dataset
    X.extend(X2)
    y.extend(y2)
