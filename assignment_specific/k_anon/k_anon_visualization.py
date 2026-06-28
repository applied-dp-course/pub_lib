import matplotlib.pyplot as plt


def make_anonymity_histogram_figure(df, title, column_to_drop, max_value=10):
    counts = (
        df.drop(columns=[column_to_drop])
        .groupby(list(df.columns.difference([column_to_drop])), observed=True)
        .size()
    )
    histogram = counts.value_counts()
    histogram = histogram.reindex(range(1, max_value + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    histogram.plot(kind='bar', ax=ax)
    ax.set_xlabel('Anonymity level')
    ax.set_ylabel('Number of equivalence classes')
    ax.set_title('Anonymity level Histogram in the ' + title + ' dataset')
    return fig
