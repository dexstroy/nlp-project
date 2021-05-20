import argparse

from random_forest import random_forest_classifier
from naive_bayes import naive_bayes_classifier
from logistic_regression import logistic_regression_classifier

classifiers = [("forest", random_forest_classifier),
               ("bayes", naive_bayes_classifier),
               ("regression", logistic_regression_classifier)]

# example of running the script
# run_classifier.py -classifier bayes -use_custom_features -use_stories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-classifier")

    # use as switch, default value is False
    parser.add_argument('-use_stories', action='store_true')
    parser.add_argument('-use_custom_features', action='store_true')
    parser.add_argument('-use_class_grouping', action='store_true')

    # Read arguments from the command line
    args = parser.parse_args()

    # Check classifier
    if args.classifier:
        found_classifier = False
        for classifier in classifiers:
            if classifier[0] == args.classifier:
                classifier[1](use_stories=args.use_stories, use_custom_features=args.use_custom_features,
                              use_class_grouping=args.use_class_grouping)
                found_classifier = True
                break
        if not found_classifier:
            print("Wrong classifier, please select one of the following options:")
            for classifier_name in classifiers:
                print(classifier_name[0])

