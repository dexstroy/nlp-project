from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from helpers.constants import ALL_CLASSES, DATASET_PATH
from helpers.worksheet import read_worksheet
from helpers.preprocess import preprocess_data
from helpers.data import get_data

if __name__ == "__main__":
    X, y = get_data()