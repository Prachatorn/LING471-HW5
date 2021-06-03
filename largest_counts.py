import sys
import pandas as pd
import operator
import matplotlib.pyplot as plt

# From Assignment 2, copied manually here just to remind you
# that you can copy stuff manually if importing isn't working out.
# You can just use this or you can replace it with your function.


def countTokens(text):
    token_counts = {}
    tokens = text.split(' ')
    for word in tokens:
        if not word in token_counts:
            token_counts[word] = 0
        token_counts[word] += 1
    return token_counts


def largest_counts(data):  # TODO: Finish implementing this function

    # TODO: Cut up the rows in the dataset according to how you stored things.
    # The below assumes test data is stored first and negative is stored before positive.
    # If you did the same, no change is required.
    #neg_test_data = data[:12500]
    pos_train_data = data[:12500]
    neg_train_data = data[12500:25000]
    pos_test_data = data[25000:37500]
    neg_test_data = data[37500:50000]
    #pos_train_data = data[37500:50000]

    # TODO: SORT the count dicts which countTokens() returns
    # by value (count) in reverse (descending) order.
    # It is your task to Google and learn how to do this, but we will help of course,
    # if you come to use with questions. This can be daunting at first, but give it time.
    # Spend some (reasonable) time across a few days if necessary, and you will do it!

    # As is, the counts returned by the counter AREN'T sorted!
    # So you won't be able to easily retrieve the most frequent words.

    # NB: str.cat() turns whole column into one text
    train_counts_pos_original = countTokens(pos_train_data["review"].str.cat())
    train_counts_pos_cleaned = countTokens(pos_train_data["cleaned_review"].str.cat())
    train_counts_pos_lowercased = countTokens(pos_train_data["lowercased"].str.cat())
    train_counts_pos_no_stop = countTokens(pos_train_data["no stopwords"].str.cat())
    train_counts_pos_lemmatized = countTokens(pos_train_data["lemmatized"].str.cat())

    train_counts_neg_original = countTokens(neg_train_data["review"].str.cat())
    train_counts_neg_cleaned = countTokens(neg_train_data["cleaned_review"].str.cat())
    train_counts_neg_lowercased = countTokens(neg_train_data["lowercased"].str.cat())
    train_counts_neg_no_stop = countTokens(neg_train_data["no stopwords"].str.cat())
    train_counts_neg_lemmatized = countTokens(neg_train_data["lemmatized"].str.cat())

    
    # https://thispointer.com/sort-a-dictionary-by-value-in-python-in-descending-ascending-order/
    sorted_dict_train_pos_original = dict(sorted(train_counts_pos_original.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_pos_cleaned = dict(sorted(train_counts_pos_cleaned.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_pos_lowercased = dict(sorted(train_counts_pos_lowercased.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_pos_no_stop = dict(sorted(train_counts_pos_no_stop.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_pos_lemmatized = dict(sorted(train_counts_pos_lemmatized.items(), key=operator.itemgetter(1), reverse=True))


    # Once the dicts are sorted, output the first 20 rows for each.
    # This is already done below, but changes may be needed depending on what you did to sort the dicts.
    # The [:19] "slicing" syntax expects a list. If you sorting call return a list (which is likely, as being sorted
    # is conceptualy a properly of LISTS,  NOT dicts),
    # you may want to remove the additional list(dict_name.items()) conversion.
    with open('counts_positive.txt', 'w') as f:
        f.write('Original POS reviews:\n')
        for k, v in list(sorted_dict_train_pos_original.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned POS reviews:\n')
        for k, v in list(sorted_dict_train_pos_cleaned.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased POS reviews:\n')
        for k, v in list(sorted_dict_train_pos_lowercased.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords POS reviews:\n')
        for k, v in list(sorted_dict_train_pos_no_stop.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized POS reviews:\n')
        for k, v in list(sorted_dict_train_pos_lemmatized.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))

        column_name = ["Word", "Count"]

        train_pos_original_list = list(sorted_dict_train_pos_original.items())[:20]
        df_train_pos_original = pd.DataFrame(train_pos_original_list, columns = column_name)
        df_train_pos_original.to_csv("train_pos_original.csv")

        train_pos_cleaned_list = list(sorted_dict_train_pos_cleaned.items())[:20]
        df_train_pos_cleaned = pd.DataFrame(train_pos_cleaned_list, columns = column_name)
        df_train_pos_cleaned.to_csv("train_pos_cleaned.csv")

        train_pos_lowercased_list = list(sorted_dict_train_pos_lowercased.items())[:20]
        df_train_pos_lowercased = pd.DataFrame(train_pos_lowercased_list, columns = column_name)
        df_train_pos_lowercased.to_csv("train_pos_lowercased.csv")

        train_pos_no_stop_list = list(sorted_dict_train_pos_no_stop.items())[:20]
        df_train_pos_no_stop = pd.DataFrame(train_pos_no_stop_list, columns = column_name)
        df_train_pos_no_stop.to_csv("train_pos_no_stop.csv")

        train_pos_lemmatized_list = list(sorted_dict_train_pos_lemmatized.items())[:20]
        df_train_pos_lemmatized = pd.DataFrame(train_pos_lemmatized_list, columns = column_name)
        df_train_pos_lemmatized.to_csv("train_pos_lemmatized.csv")

        # TODO: Do the same for all the remaining training dicts, per Assignment spec.

    sorted_dict_train_neg_original = dict(sorted(train_counts_neg_original.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_neg_cleaned = dict(sorted(train_counts_neg_cleaned.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_neg_lowercased = dict(sorted(train_counts_neg_lowercased.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_neg_no_stop = dict(sorted(train_counts_neg_no_stop.items(), key=operator.itemgetter(1), reverse=True))
    sorted_dict_train_neg_lemmatized = dict(sorted(train_counts_neg_lemmatized.items(), key=operator.itemgetter(1), reverse=True))

    with open('counts_negative.txt', 'w') as f:
        f.write('Original NEG reviews:\n')
        for k, v in list(sorted_dict_train_neg_original.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned NEG reviews:\n')
        for k, v in list(sorted_dict_train_neg_cleaned.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased NEG reviews:\n')
        for k, v in list(sorted_dict_train_neg_lowercased.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords NEG reviews:\n')
        for k, v in list(sorted_dict_train_neg_no_stop.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized NEG reviews:\n')
        for k, v in list(sorted_dict_train_neg_lemmatized.items())[:19]:
            f.write('{}\t{}\n'.format(k, v))

        column_name = ["Word", "Count"]

        train_neg_original_list = list(sorted_dict_train_neg_original.items())[:20]
        df_train_neg_original = pd.DataFrame(train_neg_original_list, columns = column_name)
        df_train_neg_original.to_csv("train_neg_original.csv")

        train_neg_cleaned_list = list(sorted_dict_train_neg_cleaned.items())[:20]
        df_train_neg_cleaned = pd.DataFrame(train_neg_cleaned_list, columns = column_name)
        df_train_neg_cleaned.to_csv("train_neg_cleaned.csv")

        train_neg_lowercased_list = list(sorted_dict_train_neg_lowercased.items())[:20]
        df_train_neg_lowercased = pd.DataFrame(train_neg_lowercased_list, columns = column_name)
        df_train_neg_lowercased.to_csv("train_neg_lowercased.csv")

        train_neg_no_stop_list = list(sorted_dict_train_neg_no_stop.items())[:20]
        df_train_neg_no_stop = pd.DataFrame(train_neg_no_stop_list, columns = column_name)
        df_train_neg_no_stop.to_csv("train_neg_no_stop.csv")

        train_neg_lemmatized_list = list(sorted_dict_train_neg_lemmatized.items())[:20]
        df_train_neg_lemmatized = pd.DataFrame(train_neg_lemmatized_list, columns = column_name)
        df_train_neg_lemmatized.to_csv("train_neg_lemmatized.csv")
    
    # TODO: Copy the output of the above print statements
    #  into your document/report, or otherwise create a table/visualization for these counts.
    # Manually is fine, or you may explore bar charts in pandas! Be creative :).

    bar_train_pos_original = df_train_pos_original.plot.bar(x="Word", y="Count")
    pict_pos_original = bar_train_pos_original.get_figure()
    plt.title("Top 20 Words With Highest Count For Positive Original")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_pos_original.savefig("bar_train_pos_original.jpeg")

    bar_train_pos_cleaned = df_train_pos_cleaned.plot.bar(x="Word", y="Count")
    pict_pos_cleaned = bar_train_pos_cleaned.get_figure()
    plt.title("Top 20 Words With Highest Count For Positive Cleaned")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_pos_cleaned.savefig("bar_train_pos_cleaned.jpeg")

    bar_train_pos_lowercased = df_train_pos_lowercased.plot.bar(x="Word", y="Count")
    pict_pos_lowercased = bar_train_pos_lowercased.get_figure()
    plt.title("Top 20 Words With Highest Count For Positive Lowercased")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_pos_lowercased.savefig("bar_train_pos_lowercased.jpeg")

    bar_train_pos_no_stop = df_train_pos_no_stop.plot.bar(x="Word", y="Count")
    pict_pos_no_stop = bar_train_pos_no_stop.get_figure()
    plt.title("Top 20 Words With Highest Count For Positive No Stop")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_pos_no_stop.savefig("bar_train_pos_no_stop.jpeg")

    bar_train_pos_lemmatized = df_train_pos_lemmatized.plot.bar(x="Word", y="Count")
    pict_pos_lemmatized = bar_train_pos_lemmatized.get_figure()
    plt.title("Top 20 Words With Highest Count For Positive Lemmatized")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_pos_lemmatized.savefig("bar_train_pos_lemmatized.jpeg")


    bar_train_neg_original = df_train_neg_original.plot.bar(x="Word", y="Count")
    pict_neg_original = bar_train_neg_original.get_figure()
    plt.title("Top 20 Words With Highest Count For Negative Original")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_neg_original.savefig("bar_train_neg_original.jpeg")

    bar_train_neg_cleaned = df_train_neg_cleaned.plot.bar(x="Word", y="Count")
    pict_neg_cleaned = bar_train_neg_cleaned.get_figure()
    plt.title("Top 20 Words With Highest Count For Negative Cleaned")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_neg_cleaned.savefig("bar_train_neg_cleaned.jpeg")

    bar_train_neg_lowercased = df_train_neg_lowercased.plot.bar(x="Word", y="Count")
    pict_neg_lowercased = bar_train_neg_lowercased.get_figure()
    plt.title("Top 20 Words With Highest Count For Negative Lowercased")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_neg_lowercased.savefig("bar_train_neg_lowercased.jpeg")

    bar_train_neg_no_stop = df_train_neg_no_stop.plot.bar(x="Word", y="Count")
    pict_neg_no_stop = bar_train_neg_no_stop.get_figure()
    plt.title("Top 20 Words With Highest Count For Negative No Stop")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_neg_no_stop.savefig("bar_train_neg_no_stop.jpeg")

    bar_train_neg_lemmatized = df_train_neg_lemmatized.plot.bar(x="Word", y="Count")
    pict_neg_lemmatized = bar_train_neg_lemmatized.get_figure()
    plt.title("Top 20 Words With Highest Count For Negative Lemmatized")
    plt.xlabel("Words")
    plt.ylabel("Counts")
    plt.tight_layout()
    pict_neg_lemmatized.savefig("bar_train_neg_lemmatized.jpeg")

def main(argv):
    data = pd.read_csv(argv[1], index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    largest_counts(data)


if __name__ == "__main__":
    main(sys.argv)
