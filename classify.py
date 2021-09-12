import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv('~/Documents/Side_Projects/binary_text_classification/nlp-getting-started/train.csv')
test_df = pd.read_csv('~/Documents/Side_Projects/binary_text_classification/nlp-getting-started/test.csv')

train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]



# feature extraction:

# number of words or common words in distaster tweets

count_vec = feature_extraction.text.CountVectorizer()


count_train_vec = count_vec.fit_transform(train_df["text"])
count_test_vec = count_vec.transform(test_df['text'])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, count_train_vec, train_df["target"], cv=3, scoring="f1")
scores
# model did okay, lets run it with Tfidf
