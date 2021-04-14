from openpyxl import load_workbook
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from expand_contractions import Expander
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


# Opens a single sheet for reading, returns a list of messages and a list of classes
def read_worksheet(filename, sheet_name, all_classes, label_encoder, no_columns):
    wb = load_workbook(filename, read_only=True)
    ws = wb[sheet_name]
    column_labels = next(ws.rows)
    X = []
    y = []
    for row in ws.rows:
        if row[0].value is None:
            break
        elif row[0].value.strip() == "Course":  # Skip the first line which only contains column titles
            continue
        new_entry = {}
        for i in range(no_columns):
            new_entry[column_labels[i].value.lower().replace(" ", "_")] = str(row[i].value)
        c_list = [new_entry["codepreliminary"].lower().strip()]
        if c_list[0] not in all_classes:
            c_list = new_entry["codepreliminary"].lower().strip().split("/")
        # If there are 2 classes listed in document add message twice (1 for each class)
        for c in c_list:
            new_entry["codepreliminary"] = label_encoder.transform([c])[0]
            X.append(new_entry["message"])
            y.append(label_encoder.transform([c])[0])
    wb.close()
    return X, y


# Iterate through every row (message) in dataset and pre-process it
def preprocess_data(dataset, classes):
    # List of emojis found in messages
    emojis = [":)", ":(", ":D", "👍"]
    ps = PorterStemmer()
    tkn = TweetTokenizer()  # Use tweet tokenizer to not split emojis as punctuation
    exp = Expander()  # Expands contractions such as I'm, he's into I am, he is
    new_dataset = []
    new_classes = []
    stopword_set = stopwords.words("english")
    for i, entry in enumerate(dataset):
        exp.expand(entry)
        new_entry = []
        # Remove capitalization, stopwords
        for token in tkn.tokenize(entry):
            new_token = ps.stem(token, True)  # Stem the token and convert to lowercase
            if new_token.startswith("http"):
                new_token = "url"  # Replace links with a url tag
            if any(emoji in new_token for emoji in emojis):
                new_token = "emoji"  # Replace emojis with a string
            if (new_token not in stopword_set or len(new_token) == 1) and new_token.isalnum():
                new_entry.append(new_token)
        if len(new_entry) > 0:
            new_dataset.append(" ".join(new_entry))
            new_classes.append(classes[i])

    return new_dataset, new_classes


if __name__ == "__main__":
    all_classes = ["content discussion", "greeting", "logistics", "assignment instructions", "instruction question",
                   "assignment question", "general comment", "response", "incomplete/typo", "feedback",
                   "emoticon/non-verbal", "discussion wrap-up", "outside material", "opening statement",
                   "general question", "content question", "general discussion"]
    le = preprocessing.LabelEncoder()
    le.fit(all_classes)  # Encode classes with numerical labels
    X, y = read_worksheet("dataset.xlsx", "Discussion only data", all_classes, le, 10)
    X2, y2 = read_worksheet("dataset.xlsx", "CREW data", all_classes, le, 11)
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

