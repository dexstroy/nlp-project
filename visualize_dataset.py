from sklearn import preprocessing
from helpers.worksheet import read_worksheet
from helpers.constants import ALL_CLASSES, DATASET_PATH
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from statistics import mean

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

def class_distributions(X, y):
    c = Counter(map(lambda x: ALL_CLASSES[x], y))
    c = OrderedDict(c.most_common())
    
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.245)
    plt.xticks(rotation=90)
    plt.title('Message class distribution')
    plt.bar(c.keys(), c.values())
    plt.xlabel('Class')
    plt.ylabel('Number of messages in class')

def average_message_length(X, y):
    class_message_lengths = {}
    for c in ALL_CLASSES:
        class_message_lengths[c] = []
    
    for idx, row in enumerate(X):
        class_message_lengths[ALL_CLASSES[y[idx]]].append(len(row['message']))

    class_message_average_lengths = {}
    for key in class_message_lengths:
        class_message_average_lengths[key] = mean(class_message_lengths[key])
    
    c = OrderedDict(sorted(class_message_average_lengths.items(), reverse=True, key=lambda x: x[1]))

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.245)
    plt.xticks(rotation=90)
    plt.title('Average length of messages in all classes')
    plt.bar(c.keys(), c.values())
    plt.xlabel('Class')
    plt.ylabel('Message length')
    

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
    class_distributions(X, y)
    average_message_length(X, y)

    # show visualizations
    plt.show()
