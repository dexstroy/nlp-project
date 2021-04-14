import re

contractions = [
    (r"won\'t", "will not"),
    (r"can\'t", "cannot"),
    (r"i\'m", "i am"),
    (r"ain\'t", "is not"),
    (r"(\w+)\'ll", "\g<1> will"),
    (r"(\w+)n\'t", "\g<1> not"),
    (r"(\w+)\'ve", "\g<1> have"),
    (r"(\w+)\'s", "\g<1> is"),
    (r"(\w+)\'re", "\g<1> are"),
    (r"(\w+)\'d", "\g<1> would")
]


class Expander(object):
    def __init__(self, patterns=contractions):
        self.patterns = [(re.compile(regex, re.IGNORECASE), repl) for (regex, repl) in patterns]

    def expand(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s