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
    "general_conversation",
    "task_discussion",
    "directly_related_content"
]

GROUP_MAPPER ={
    "opening statement": "general_conversation",
    "general discussion": "general_conversation",
    "greeting": "general_conversation",
    "emoticon/non-verbal": "general_conversation",
    "incomplete/typo": "general_conversation",
    "general comment": "general_conversation",
    "discussion wrap-up": "general_conversation",
    "general question": "general_conversation",
    "feedback": "general_conversation",
    "outside material": "general_conversation",

    "logistics": "task_discussion",
    "assignment instructions": "task_discussion",
    "instruction question": "task_discussion",
    "assignment question": "task_discussion",

    "content question": "directly_related_content",
    "content discussion": "directly_related_content",
    "response": "directly_related_content"
}