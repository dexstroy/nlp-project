
# IMapBook collaborative discussions classification

## Authors
- Matej Rus (63160280)
- Nik Zupančič (63150325)
- Aleks Vujić (63160333)

## How to run
### Common steps
1. Clone the repository
2. Run `pip install -r requirements.txt` to install all dependencies

### Classification of the messages 
To classify the messages with traditional classifications techniques (random forest, naive Bayes, logistic regression), run the classifier with: `python run_classifier.py -classifier <classifier_name> -use_custom_features -use_stories -use_class_grouping`
1. `<classifier_name>` can be one of: `forest`, `bayes`, `regression`
2. `-use_custom_features` is *optional* flag which indicates the use of custom features (number of URLs, number of punctuations, number of emojis, ...)
3. `-use_stories` is *optional* flag which indicates that stories are also used to help with classification. Content of the stories is treated as *content discussion* class.
4. `-use_class_grouping` is *optional* flag which indicates that classes are grouped. Look at `helper/constants.py` to see default class grouping.

To classify the messages with BERT model, run `bert.ipynb` on local server or access it in [Google Colab](https://colab.research.google.com/drive/1leHD3ptQg8NOd-YoN4FGleYOKCwEZ8CL?usp=sharing). Be sure to run it on **GPU** (navigate to Edit->Notebook Settings and select GPU from the Hardware Acceleration dropdown). When running on Google Colab, upload the dataset in the same folder as the notebook file.
If you're running it locally the variable `DATASET_PATH` should point to the dataset inside `data` folder (uncomment the line `DATASET_PATH = "data/dataset.xlsx"`).


### Visualization of the dataset
Run `python visualize_dataset.py`.

## Repository structure
- folder `.vscode` contains debug configuration if project is run using Visual Studio Code
- folder `data` contains dataset and stories
- folder `helpers` contains utility classes and functions (tokenization, preprocessing, worksheet helpers, constants, ...)
- folder `report` contains final report
- `bert.ipynb` is Python notebook which contains code for running BERT model locally or in Google Colab
- `logistic_regression.py` contains code for classifying messages with logistic regression
- `naive_bayes.py` contains code for classifying messages with naive Bayes
- `random_forest.py` contains code for classifying messages with random forest
- `run_classifier.py` is main file used to trigger any method of classification (logistic regression, naive Bayes and random forest)
- `visualize_dataset.py` is used to draw plots of basic dataset statistics
