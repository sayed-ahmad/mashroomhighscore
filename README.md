# mushroomhighscore

This project builds a machine learning model that classifies mushrooms as:

- **edible**
- **poisonous**

based on their physical characteristics (for example cap shape, odor, and gill
size). This is a classic supervised **binary classification** task.

## Run the classifier

The repository includes a lightweight Naive Bayes implementation in pure Python:

```bash
python mushroom_classifier.py
```

You can also train/evaluate using your own CSV file:

```bash
python mushroom_classifier.py --data /path/to/mushrooms.csv --label-column class
```
