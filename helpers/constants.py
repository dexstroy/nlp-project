DATASET_PATH = "data/dataset.xlsx"
'''
ALL_CLASSES = [
    "content discussion", "greeting", "logistics", "assignment instructions", "instruction question",
    "assignment question", "general comment", "response", "incomplete/typo", "feedback",
    "emoticon/non-verbal", "discussion wrap-up", "outside material", "opening statement",
    "general question", "content question", "general discussion"
]

'''

ALL_CLASSES = [
    "content discussion", "greeting", "logistics", "assignment instructions",
    "general comment", "response", "incomplete/typo", "feedback",
    "emoticon/non-verbal", "discussion wrap-up", "outside material", "opening statement",
    "general discussion", "questions"
]

GROUP_MAPPER ={
    "instruction question": "questions",
    "assignment question": "questions",
    "general question": "questions",
    "content question": "questions"
}