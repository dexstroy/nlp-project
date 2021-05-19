from sklearn import preprocessing
from helpers.worksheet import read_worksheet
from helpers.constants import ALL_CLASSES, DATASET_PATH
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

def get_dataset():
    # initialize label encoder
    le = preprocessing.LabelEncoder()
    le.fit(ALL_CLASSES)

    # load worksheets - include all columns
    X, y = read_worksheet(DATASET_PATH, "Discussion only data", ALL_CLASSES, le, 10, only_messages=False)
    X2, y2 = read_worksheet(DATASET_PATH, "CREW data", ALL_CLASSES, le, 11, only_messages=False)
    
    # join worksheets into a single dataset
    X.extend(X2)
    y.extend(y2)

    return X, y

def most_active_students(X, y):
    c = Counter(map(lambda x: x['pseudonym'], X))
    c = OrderedDict(c.most_common(10))

    plt.figure()
    plt.title('Students with most number of messages')
    plt.bar(c.keys(), c.values())
    plt.xlabel('Student pseudonym')
    plt.ylabel('Number of sent messages')

if __name__ == "__main__":
    # configure plot settings
    plt.rcParams.update({
        'axes.titlesize': 'xx-large',
        'axes.labelsize': 'x-large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large'
    })

    # get dataset
    X, y = get_dataset()

    # prepare visualizations
    most_active_students(X, y)

    # show visualizations
    plt.show()
