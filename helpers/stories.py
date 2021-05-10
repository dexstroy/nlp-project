# reads stories and classifies them as content discussion
def read_stories(stories_path, label_encoder):
    X = []
    y = []
    cd_code = label_encoder.transform(["content discussion"])[0]

    for story_path in stories_path:
        story = open(story_path, 'r')
        lines = story.readlines()

        for line in lines:
            X.append(line.replace("\n", ""))
            y.append(cd_code)

    return X, y