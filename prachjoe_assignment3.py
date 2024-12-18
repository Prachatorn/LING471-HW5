# This is a skeleton for your assignment 2 program.
# It contains the program structure, function names,
# the main function already written,
# and comment instructions regarding what you should write.

'''
SUMMARY OF WHAT YOU NEED TO DO:
You will only need to fill in several lines of code.
The program looks long because of the comments; it is actually very short.
Look for TODO items! (But read the comments, too :) ).
'''

# Import the system module
import sys

# Import regular expressions module
import re

# Import the string module to access its punctuation set
import string

# Import the iteration over a directory for many files.
import os

# Import review_vector.py property
# import review_vector

# Import evaluation.py property
import evaluation

'''
The below function should be called on a file name.
It should open the file, read its contents, and store it in a variable.
Then it should remove punctuation marks and return the "cleaned" text.
'''


def cleanFileContents(f):
    # The below two lines open the file and read all the text from it
    # storing it into a variable called "text".
    # You do not need to modify the below two lines; they are already working as needed.
    with open(f, 'r', encoding="utf8") as f:
        text = f.read()

    # The below line will clean the text of punctuation marks.
    # Ask if you are curious about how it works! But you can just use it as is.
    # Observe the effect of the function by inspecting the debugger pane while stepping over.
    clean_text = text.translate(str.maketrans('', '', string.punctuation))

    # Now, we will want to replace all tabs with spaces, and also all occurrences of more than one
    # space in a row with a single space. Review the regular expression slides/readings, and
    # write a statement below which replaces all occurrences of one or more whitespace-group characters
    # (that will include tabs) with a single space. You want the re.sub function.
    # The shortcut for all whitespace characters is \s. The regex operator for "one or more" is +.
    # Read re.sub()'s documentation to understand which argument goes where in the parentheses.

    # TODO: Your call to the re.sub function of the regular expression module here.
    # As is, the value of clean_text does not change.
    # clean_text = clean_text
    clean_text = re.sub(pattern=r"\s+", repl=r" ", string=clean_text)

    # Do not forget to return the result!
    return clean_text


'''
The below function takes a string as input, breaks it down into word tokens by space, and stores, in a dictionary table,
how many times each word occurred in the text. It returns the dictionary table.
'''


def countTokens(text):
    # Initializing an empty dictionary. Do not modify the line below, it is already doing what is needed.
    token_counts = {}

    # Use the split() function, defined for strings, to split the text by space.
    # Store the result in a variable, e.g. called "tokens".
    # See what the split() function returns and stores in your variable
    # as you step through the execution in the debugger.

    # TODO: Write a statement below calling split() on your text and storing the
    # result in a new variable.

    tokens = text.split()

    # Now, we need to iterate over each word in the list of tokens
    # (write a for loop over the list that split() returned).
    # Inside the loop, so, for each word, we will perform some conditional logic:
    # If the word is not yet stored in the dictionary
    # we called "token_counts" as a key, we will store it there now,
    # and we will initialize the key's value to 0.
    # Outside that if statement: now that we are sure
    # the word is stored as a key, we will increment the count by 1.

    # TODO: Write a for loop here, doing what is described above.

    for x in tokens:
        if x not in token_counts:
            token_counts[x] = 1
        else:
            token_counts[x] += 1

    # Do not forget to return the result!
    return token_counts

def simplePrediction(counts):
    # This line retrieves the count for "good". If the word "good" is not found in "counts", it returns 0.
    pos_count = counts.get(POS, 0)
    # TODO: Write a similar statement below to retrieve the count of "bad".
    neg_count = counts.get(NEG, 0)

    # TODO: Write an if-elif-else block here, following the logic described in the function description.
    # Do not forget to return the prediction! You will be returning one of the constants declared above.
    # You may choose to store a prediction in a variable and then write the return statement outside
    # of the if-else block, or you can have three return statements within the if-else block.

    if pos_count > neg_count:
        return POS_REVIEW
    elif neg_count > pos_count:
        return NEG_REVIEW
    else:
    # TODO: You will modify the below return statement or move it into your if-else block when you write it.
        return NONE


'''
This silly "prediction funtion" will do the following "rudimentary data science":
If a review contains more of the word "good" than of the word "bad", 
the function predicts "positive" (by returning a string "POSITIVE").
If it contains more of the word "bad" than of the word "good",
the function predicts "negative". 
If the count is equal (note that this includes zero count),
the function cannot make a prediction and returns a string "NONE".
'''

# Constants. Constants are important to avoid typo-related bugs, among other reasons.
# Use these constants as return values for the below function.

POS_REVIEW = "POSITIVE"
NEG_REVIEW = "NEGATIVE"
NONE = "NONE"
POS = 'good'
NEG = 'bad'


def predictSimplistic(filename):

    # The file that you will read should be passed as the argument to the program.
    # From python's point of view, it is the element number 1 in the array called argv.
    # argv is a special variable name. We don't define it ourselves; python knows about it.
    # Place the first breakpoint here, when starting.


    # Now, we will call a function called cleanFileContents on the filename we were passed.
    # NB: We could have called the function directly on argv[1]; that would have the same effect.
    clean_text = cleanFileContents(filename)

    # Now, we will count how many times each word occurs in the review.
    # We are passing the text of the review, cleaned from punctuation, to the function called countTokens.
    # We assign the output of the function to a new variable we call tokens_with_counts.
    tokens_with_counts = countTokens(clean_text)

    # Call the simplistic prediction function on the obtained counts.
    # Store the output of the function in a new variable called "prediction".
    prediction = simplePrediction(tokens_with_counts)

    # Finally, let's print out what we predicted. Note how we are calling the format()
    # function on the string we are printing out, and we are passing it two
    # arguments: the file name and our prediction. This is a convenient way of
    # printing out results. We will keep using it in the future.

    #print("The prediction for file {} is {}".format(filename, prediction))

    return prediction

'''The main function is the entry point of the program.
When debugging, if you want to start from the very beginning,
start here. NB: Put the breakpoint not on the "def" line but below it.
Do not modify this function; we already wrote it for you.
You need to modify the functions which it calls, not the main() itself.
'''

def main(argv):
    dirname_pos = argv[1]
    review_vecs_pos_text = []
    review_pos_gold_label = []
    review_vecs_pos = []
    review_pos_prediction = []
    for filename in os.listdir(dirname_pos):
        f = os.path.join(dirname_pos, filename)
        if os.path.isfile(f) and filename.endswith('.txt'):
            review_pos_prediction.append(predictSimplistic(f))
            # Used for Part 4 of the assignment
            #review_vecs_pos_text.append(cleanFileContents(f))
            review_pos_gold_label.append(POS_REVIEW)
    # Used for Part 4 of the assignment
    #for x in range(0, len(review_vecs_pos_text)):
        #review_vecs_pos.append(review_vector.reviewVec(review_vecs_pos_text[x], review_pos_gold_label[x]))

    dirname_neg = argv[2]
    review_vecs_neg_text = []
    review_neg_gold_label = []
    review_vecs_neg = []
    review_neg_prediction = []
    for filename in os.listdir(dirname_neg):
        f = os.path.join(dirname_neg, filename)
        if os.path.isfile(f) and filename.endswith('.txt'):
            review_neg_prediction.append(predictSimplistic(f))
            # Used for Part 4 of the assignment
            #review_vecs_neg_text.append(cleanFileContents(f))
            review_neg_gold_label.append(NEG_REVIEW)
    # Used for Part 4 of the assignment
    #for x in range(0, len(review_vecs_neg_text)):
    #    review_vecs_neg.append(review_vector.reviewVec(review_vecs_neg_text[x], review_neg_gold_label[x]))

    total_prediction = review_pos_prediction + review_neg_prediction
    total_gold_label = review_pos_gold_label + review_neg_gold_label

    accuracy_prediction = evaluation.computeAccuracy(total_prediction, total_gold_label)
    accuracy_prediction = round(accuracy_prediction[0], 4)
    #print("The accuracy of Simplistic Prediction is " + str(accuracy_prediction))
    print(accuracy_prediction)
    precision_pos_prediction = evaluation.computePrecisionRecall(total_prediction, total_gold_label, POS_REVIEW)
    pos_precision = round(precision_pos_prediction[0], 4)
    pos_recall = round(precision_pos_prediction[1], 4)
    print(pos_precision)
    print(pos_recall)
    #print("Pos: Precision: " + str(pos_precision) + "; Recall " + str(pos_recall))
    precision_neg_prediction = evaluation.computePrecisionRecall(total_prediction, total_gold_label, NEG_REVIEW)
    neg_precision = round(precision_neg_prediction[0], 4)
    neg_recall = round(precision_neg_prediction[1], 4)
    #print("Neg: Precision: " + str(neg_precision) + "; Recall " + str(neg_recall))
    print(neg_precision)
    print(neg_recall)

# The below code is needed so that this file can be used as a module.
# If we want to call our program from outside this window, in other words.
if __name__ == "__main__":
    main(sys.argv)
