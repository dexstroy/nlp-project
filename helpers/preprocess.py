from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from helpers.expand_contractions import Expander


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