import matplotlib.pyplot as plt


def plot_one_run_auditing_experiement(df):
    df = df.round(4)
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines connecting lower_eps and upper_eps for each acc_synthetic_train
    for idx, row in df.iterrows():
        # Generate a unique color for this row
        color = plt.cm.tab10(idx % 10)

        # Plot the line with the row's color
        ax.plot(
            [row['acc_labeled_test'], row['acc_labeled_test']],
            [row['lower_eps'], row['upper_eps']],
            color=color,
            label=f"Acc Train: {row['acc_labeled_train']}, Acc Test: {row['acc_labeled_test']}",
        )

        # Add markers with the same color
        ax.scatter(row['acc_labeled_test'], row['lower_eps'], marker='^', s=100, color=color)
        ax.scatter(row['acc_labeled_test'], row['upper_eps'], marker='v', s=100, color=color)

        # Add y-values next to the points
        ax.text(
            row['acc_labeled_test'],
            row['lower_eps'],
            f"{row['lower_eps']:.3f}",
            fontsize=12,
            color='red',
            ha='right',
            va='bottom',
        )
        ax.text(
            row['acc_labeled_test'],
            row['upper_eps'],
            f"{row['upper_eps']:.3f}",
            fontsize=12,
            color='red',
            ha='right',
            va='top',
        )

    ax.set_xlabel('Accuracy on Test Data')
    ax.set_ylabel('Epsilon Values')
    ax.set_title('Epsilon Ranges for Different Train Accuracies')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True)

    # Add a table below the plot with specific parameters
    table_data = df[
        ['noise_factor', 'subset_size', 'num_epochs', 'batch_size', 'lower_eps', 'upper_eps']
    ].to_dict('records')
    table_columns = [
        'Noise Factor',
        'Training data Size',
        'Num Epochs',
        'Batch Size',
        'Lower ε',
        'Upper ε',
    ]
    table = plt.table(
        cellText=[list(row.values()) for row in table_data],
        colLabels=table_columns,
        cellLoc='center',
        loc='bottom',
        bbox=[0.0, -0.6, 1.0, 0.3],
        fontsize=8,
    )

    # Adjust layout to accommodate the table
    plt.subplots_adjust(bottom=0.4, top=1.3)

    plt.show()
