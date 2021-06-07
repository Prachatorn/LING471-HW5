import pandas as pd
import string
import os
import sys

# sklearn is installed via pip: pip install -U scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt

# TODO: Your custom imports here; or copy the functions to here manually.
from evaluation import computeAccuracy, computePrecisionRecall
from prachjoe_assignment3 import predictSimplistic

# TODO: You may need to modify assignment 4 if you just had a main() there.
# my_naive_bayes() should take a column as input and return as output 10 floats (numbers)
# representing the metrics.
from prachjoe_naive_bayes import my_naive_bayes


def main(argv):
    data = pd.read_csv('prachjoe_imdb_expanded.csv', index_col=[0])
    #print(data.head())  # <- Verify the format. Comment this back out once done.

    # Part II:
    # Run all models and store the results in variables (dicts).
    # TODO: Make sure you imported your own naive bayes function and it works properly with a named column input!
    # TODO: See also the next todo which gives an example of a convenient output for my_naive_bayes()
    # which you can then easily use to collect different scores.
    # For example (and as illustrated below), the models (nb_original, nb_cleaned, etc.) can be not just lists of scores
    # but dicts where each score will be stored by key, like [TEST][POS][RECALL], etc.
    # But you can also just use lists, except then you must not make a mistake, which score you are accessing,
    # when you plot graphs.
    nb_original = my_naive_bayes(data['review'], data['label'])
    nb_cleaned = my_naive_bayes(data['cleaned_review'], data['label'])
    nb_lowercase = my_naive_bayes(data['lowercased'], data['label'])
    nb_no_stop = my_naive_bayes(data['no stopwords'], data['label'])
    nb_lemmatized = my_naive_bayes(data['lemmatized'], data['label'])

    # Collect accuracies and other scores across models.
    # TODO: Harmonize this with your own naive_bayes() function!
    # The below assumes that naive_bayes() returns a fairly complex dict of scores.
    # (NB: The dicts there contain other dicts!)
    # The return statement for that function looks like this:
    # return({'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train}, 'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}}, 'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test}, 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}})
    # This of course assumes that variables like "accuracy_train", etc., were assigned the right values already.
    # You don't have to do it this way; we are giving it to you just as an example.
    train_accuracies = []
    train_pos_precision = []
    train_pos_recall = []
    train_neg_precision = []
    train_neg_recall = []
    test_accuracies = []
    test_pos_precision = []
    test_pos_recall = []
    test_neg_precision = []
    test_neg_recall = []
    # TODO: Initialize other score lists similarly. The precision and recalls, for negative and positive, train and test.
    different_models = [nb_original, nb_cleaned, nb_lowercase, nb_no_stop, nb_lemmatized]
    for model in different_models:
        # TODO: See comment above about where this "model" dict comes from.
        # If you are doing something different, e.g. just a list of scores,
        # that's fine, change the below as appropriate,
        # just make sure you don't confuse where which score is.
        train_accuracies.append(model['Train']['accuracy'])
        train_pos_precision.append(model['Train']['POS']['precision'])
        train_pos_recall.append(model['Train']['POS']['recall'])
        train_neg_precision.append(model['Train']['NEG']['precision'])
        train_neg_recall.append(model['Train']['NEG']['recall'])
        test_accuracies.append(model['Test']['accuracy'])
        test_pos_precision.append(model['Test']['POS']['precision'])
        test_pos_recall.append(model['Test']['POS']['recall'])
        test_neg_precision.append(model['Test']['NEG']['precision'])
        test_neg_recall.append(model['Test']['NEG']['recall'])
        # TODO: Collect other scores similarly. The precision and recalls, for negative and positive, train and test.

    # TODO: Create the plot(s) that you want for the report using matplotlib (plt).
    # Use the below to save pictures as files:
    column_names = ['Original', 'Cleaned', 'Lowercased', 'No Stop', 'Lemmatized']
    x_axis = np.arange(len(column_names))
    width = 0.3

    plt.bar(x_axis, train_accuracies, width, color="Green", label="Train")
    plt.bar(x_axis + 0.3, test_accuracies, width, color="Orange", label="Test")
    plt.title('Accuracies For Different Models')
    plt.xlabel('Models')
    plt.ylabel('Percentage Score (%)')
    plt.xticks(x_axis, column_names)
    #plt.legend(loc='best')
    plt.legend(loc='lower right')    
    plt.savefig('accuracies.png')
    plt.clf()

    plt.bar(x_axis, train_pos_precision, width, color="Blue", label="Positive")
    plt.bar(x_axis + 0.3, train_neg_precision, width, color="Red", label="Negative")
    plt.title('Train Precision For Different Models')
    plt.xlabel('Models')
    plt.ylabel('Percentage Score (%)')
    plt.xticks(x_axis, column_names)
    plt.legend(loc='lower right')   
    #plt.legend(loc='best')
    plt.savefig('train_precision.png')
    plt.clf()

    plt.bar(x_axis, train_pos_recall, width, color="Blue", label="Positive")
    plt.bar(x_axis + 0.3, train_neg_recall, width, color="Red", label="Negative")
    plt.title('Train Recall For Different Models')
    plt.xlabel('Models')
    plt.ylabel('Percentage Score (%)')
    plt.xticks(x_axis, column_names)
    #plt.legend(loc='best')
    plt.legend(loc='lower right')
    plt.savefig('train_recall.png')
    plt.clf()

    #plt.bar(column_names, train_neg_precision)
    #plt.title('Train Negative Precision For Different Models')
    #plt.xlabel('Models')
    #plt.ylabel('Precision (%)')
    #plt.savefig('train_neg_precision.png')

    #plt.bar(column_names, train_neg_recall)
    #plt.title('Train Negative Recall For Different Models')
    #plt.xlabel('Models')
    #plt.ylabel('Recall (%)')
    #plt.savefig('train_neg_recall.png')

    #plt.bar(column_names, test_accuracies)
    #plt.title('Test Accuracies For Different Models')
    #plt.xlabel('Models')
    #plt.ylabel('Accuracies (%)')
    #plt.savefig('test_accuracies.png')

    plt.bar(x_axis, test_pos_precision, width, color="Blue", label="Positive")
    plt.bar(x_axis + 0.3, test_neg_precision, width, color="Red", label="Negative")
    plt.title('Test Precision For Different Models')
    plt.xlabel('Models')
    plt.ylabel('Percentage Score (%)')
    plt.xticks(x_axis, column_names)
    #plt.legend(loc='best')
    plt.legend(loc='lower right')
    plt.savefig('test_precision.png')
    plt.clf()

    plt.bar(x_axis, test_pos_recall, width, color="Blue", label="Positive")
    plt.bar(x_axis + 0.3, test_neg_recall, width, color="Red", label="Negative")
    plt.title('Test Recall For Different Models')
    plt.xlabel('Models')
    plt.ylabel('Percentage Score (%)')
    plt.xticks(x_axis, column_names)
    #plt.legend(loc='best')
    plt.legend(loc='lower right')
    plt.savefig('test_recall.png')
    plt.clf()

    #plt.bar(column_names, test_neg_precision)
    #plt.title('Test Negative Precision For Different Models')
    #plt.xlabel('Models')
    #plt.ylabel('Precision (%)')
    #plt.savefig('test_neg_precision.png')

    #plt.bar(column_names, test_neg_recall)
    #plt.title('Test Negative Recall For Different Models')
    #plt.xlabel('Models')
    #plt.ylabel('Recall (%)')
    #plt.savefig('test_neg_recall.png')

if __name__ == "__main__":
    main(sys.argv)
