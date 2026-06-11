import matplotlib.pyplot as plt


def plot_histogram(df, title, column_to_drop, max_value=10):
    '''
    Plot the histogram of the anonymity levels of the equivalence classes in the dataset, where an equivilence class is a unique row (discarding the sensitive attribute) and the anonymity level is the number of times this equivalence class appears in the dataset.
    :param df: the dataset
    :param title: the title of the plot
    :param column_to_drop: the sensitive attribute
    :param max_value: the maximum value of the anonymity levels
    :return: None
    '''
    counts = (
        df.drop(columns=[column_to_drop])
        .groupby(list(df.columns.difference([column_to_drop])), observed=True)
        .size()
    )
    histogram = counts.value_counts()
    histogram = histogram.reindex(range(1, max_value + 1), fill_value=0)
    plt.figure(figsize=(10, 6))
    histogram.plot(kind='bar')
    plt.xlabel('Anonymity level')
    plt.ylabel('Number of equivalence classes')
    plt.title('Anonymity level Histogram in the ' + title + ' dataset')
    plt.show()
