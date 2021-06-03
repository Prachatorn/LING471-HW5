# Skeleton for Assignment 4.
# Ling471 Spring 2021.

import pandas as pd
import string
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# These are your own functions you wrote for Assignment 3:
from evaluation import computePrecisionRecall, computeAccuracy


# Constants
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1


# This function will be reporting errors due to variables which were not assigned any value.
# Your task is to get it working! You can comment out things which aren't working at first.
def my_naive_bayes(argv):

    # Read in the data. NB: You may get an extra Unnamed column with indices; this is OK.
    # If you like, you can get rid of it by passing a second argument to the read_csv(): index_col=[0].
    data = pd.read_csv(argv[1], index_col=[0])
    #print(data.head()) # <- Verify the format. Comment this back out once done.

    # TODO: Change as appropriate, if you stored data differently (e.g. if you put train data first).
    # You may also make use of the "type" column here instead! E.g. you could sort data by "type".
    # At any rate, make sure you are grabbing the right data! Double check with temporary print statements,
    # e.g. print(test_data.head()).

    test_data = data[:25000]  # Assuming the first 25,000 rows are test data.

    # Assuming the second 25,000 rows are training data. Double check!
    train_data = data[25000:50000]

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by the appropriate column names.
    X_train = train_data["review"]
    y_train = train_data["label"]
    X_test = test_data["review"]
    y_test = test_data["label"]

    # TODO COMMENT: Look up what the astype() method is doing and add a comment, explaining in your own words,
    # what the next two lines are doing.
    # The astype() method is coverting an existing column to a categorical type, like integers or floats. 
    # In this case, the next two lines are converting the labels column in the .csv file into integers. 
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # TODO COMMENT: Look up what TfidfVectorizer is and what its methods "fit_transform" and "transform" are doing,
    # and add a comment, explaining in your own words, what the next three lines are doing.
    
    # The TfidfVectorizer() method transforms a collection of your corpus, or text documents, to a matrix of TF-IDF
    # features, and it seems to accept what type of ngram as a parameter (in our case, unigram and bigram).
    # The fit_transform() method learns the vocabulary and idf of a corpus and returns a document-term matrix, and 
    # takes in a collection of raw documents as a parameter, in our case the reviews from the train dataset.
    # The transform() method, similar to fit_transform() method, transforms documents to document-term matrix, but it
    # does not learn the vocaublary and idf, and it takes in a collection of raw documents as a parameter, which in our
    # case would be the reviews from the test dataset.
    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    # TODO COMMENT: The hyperparameter alpha is used for Laplace Smoothing.
    # Add a brief comment, trying to explain, in your own words, what smoothing is for.
    
    # The MultinomialNB() method is used for classification with discrete features and requires
    # integer feature counts or frantional counts like TF-IDF. This method accepts an "alpha" parameter,
    # which we set it equal to 1, meaning that it is an addtiive Laplace smoothing parameter.
    clf = MultinomialNB(alpha=ALPHA)

    # TODO COMMENT: Add a comment explaining in your own words what the "fit" method is doing.
    
    # The fit() method learns the vocabulary and IDF from a training set, which takes in a collection of
    # raw documents and an integer feature counts as a parameter, which in our case our the reviews and 
    # the labels for them in our training dataset, respectively.
    clf.fit(tf_idf_train, y_train)

    # TODO COMMENT: Add a comment explaining in your own words what the "predict" method is doing in the next two lines.

    # The predict() method predicts the labels of the data, based on the training model you used and takes in a collection
    # of raw documents as a parameter, which in our case are the reviews from both the training and testing dataset. This 
    # means that this method will predict the labels from both the training and the testing documents. 
    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    # TODO: Compute accuracy, precision, and recall, for both train and test data.
    # Import and call your methods from evaluation.py which you wrote for HW2.
    # NB: If you methods there accept lists, you may need to cast your pandas label objects to simple python lists:
    # e.g. list(y_train) -- when passing them to your accuracy and precision and recall functions.

    accuracy_test = computeAccuracy(list(y_pred_test), list(y_test))
    accuracy_train = computeAccuracy(list(y_pred_train), list(y_train))
    precision_pos_test, recall_pos_test = computePrecisionRecall(list(y_pred_test), list(y_test), GOOD_REVIEW)
    precision_neg_test, recall_neg_test = computePrecisionRecall(list(y_pred_test), list(y_test), BAD_REVIEW)
    precision_pos_train, recall_pos_train = computePrecisionRecall(list(y_pred_train), list(y_train), GOOD_REVIEW)
    precision_neg_train, recall_neg_train = computePrecisionRecall(list(y_pred_train), list(y_train), BAD_REVIEW)

    # Report the metrics via standard output.
    # Please DO NOT modify the format (for grading purposes).
    # You may change the variable names of course, if you used different ones above.

    print("Train accuracy:           \t{}".format(round(accuracy_train[0], ROUND)))
    print("Train precision positive: \t{}".format(round(precision_pos_train, ROUND)))
    print("Train recall positive:    \t{}".format(round(recall_pos_train, ROUND)))
    print("Train precision negative: \t{}".format(round(precision_neg_train, ROUND)))
    print("Train recall negative:    \t{}".format(round(recall_neg_train, ROUND)))
    print("Test accuracy:            \t{}".format(round(accuracy_test[0], ROUND)))
    print("Test precision positive:  \t{}".format(round(precision_pos_test, ROUND)))
    print("Test recall positive:     \t{}".format(round(recall_pos_test, ROUND)))
    print("Test precision negative:  \t{}".format(round(precision_neg_test, ROUND)))
    print("Test recall negative:     \t{}".format(round(recall_neg_test, ROUND)))


#if __name__ == "__main__":
#    main(sys.argv)
