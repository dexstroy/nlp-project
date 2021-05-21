from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from helpers.expand_contractions import Expander
import re


def count_url(message):
    return message.count("http")


def count_punct(message, punct):
    return message.count(punct)


def count_emoji(message):
    emojis = [":)", ":(", ":D", "👍"]
    counter = 0
    for e in emojis:
        counter += message.count(e)
    return counter


def count_interrogative_words(message):
    pronouns = ["who", "what", "when", "where", "why", "how", "when"]
    counter = 0
    for p in pronouns:
        counter += message.count(p)
    return counter


# Iterate through every row (message) in dataset and extract custom features
def get_custom_features(dataset, classes):
    # List of emojis found in messages
    emojis = [":)", ":(", ":D", "👍"]
    ps = PorterStemmer()
    tkn = TweetTokenizer()  # Use tweet tokenizer to not split emojis as punctuation
    exp = Expander()  # Expands contractions such as I'm, he's into I am, he is
    new_dataset = []
    new_classes = []
    stopword_set = stopwords.words("english")
    for i, entry in enumerate(dataset):
        entry, contractions = exp.expand(entry)
        new_entry = []
        features = []
        tokens = tkn.tokenize(entry)

        lower_entry = entry.lower()

        # Number of characters in a message
        features.append(len(entry))
        # Number of tokens in a message
        features.append(len(tokens))
        # Number of punctuations in each message
        features.append(count_punct(entry, "!") / features[0])
        features.append(count_punct(entry, "?") / features[0])
        features.append(count_punct(entry, ".") / features[0])
        # Number of URLs in a message and normalize with number of tokens
        features.append(count_url(lower_entry) / features[1])
        # Number of question pronouns: who, what... and normalize
        features.append(count_interrogative_words(lower_entry) / features[1])
        features.append(count_emoji(lower_entry) / features[0])
        features.append(contractions)

        # Remove capitalization, stopwords
        for token in tokens:
            token = token.lower()
            new_token = ps.stem(token)  # Stem the token
            if new_token.startswith("http"):
                new_token = "url"  # Replace links with a url tag
            if any(emoji in new_token for emoji in emojis):
                new_token = "emoji"  # Replace emojis with a string
            if (new_token not in stopword_set or len(new_token) == 1) and new_token.isalnum():
                new_entry.append(new_token)
        if len(new_entry) > 0:
            #new_dataset.append(" ".join(new_entry))
            new_dataset.append((" ".join(new_entry), features))
            new_classes.append(classes[i])

    return new_dataset, new_classes
