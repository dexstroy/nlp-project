# IMapBook collaborative discussions classification

## Authors
- Matej Rus (63160280)
- Nik Zupančič (63150325)
- Aleks Vujić (63160333)

## How to run
1. Clone the repository
2. Run `pip install -r requirements.txt` to install all dependencies
3. Run the classifier with: `python run_classifier.py -classifier <classifier_name> -use_custom_features -use_stories`
    1. `<classifier_name>` can be one of: `forest`, `bayes`, `regression`
    2. `-use_custom_features` is *optional* flag which indicates the use of custom features (number of URLs, number of punctuations, number of emojis, ...)
    3. `-use_stories` is *optional* flag which indicates that stories are also used in classification. Content of the stories is treated as *content discussion* class.