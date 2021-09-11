import pandas as pd
import numpy as np
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv('~/Documents/Side_Projects/binary_text_classification/nlp-getting-started/train.csv')
test_df = pd.read_csv('~/Documents/Side_Projects/binary_text_classification/nlp-getting-started/test.csv')

train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]




