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
To classify the messages with traditional classifications techniques (random forst, naive Bayes, logistic regression), run the classifier with: `python run_classifier.py -classifier <classifier_name> -use_custom_features -use_stories`
1. `<classifier_name>` can be one of: `forest`, `bayes`, `regression`
2. `-use_custom_features` is *optional* flag which indicates the use of custom features (number of URLs, number of punctuations, number of emojis, ...)
3. `-use_stories` is *optional* flag which indicates that stories are also used to help with classification. Content of the stories is treated as *content discussion* class.

To classify the messages with BERT model, run `bert.ipynb` on local server or access it in [Google Colab](https://colab.research.google.com/drive/1leHD3ptQg8NOd-YoN4FGleYOKCwEZ8CL?usp=sharing). Be sure to run it on **GPU** (navigate to Edit->Notebook Settings and select GPU from the Hardware Acceleration dropdown).

### Visualization of the dataset
Run `python visualize_dataset.py`.

## Repository structure
TODO