import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import markdown as md

directory = "data/"
dataFileNames = [
    directory + "train_data_1.txt",
    directory + "train_data_2.txt",
    directory + "train_data_3.txt",
    directory + "test_data_4.txt",
]
reportFileName = "report3.md"


def save_markdown_report(file, arr):
    for block in arr:
        file.write(block)


def display_plot_of_temperatures():
    df1 = pd.read_csv(dataFileNames[0], header=None)
    df2 = pd.read_csv(dataFileNames[1], header=None)
    df3 = pd.read_csv(dataFileNames[2], header=None)
    df4 = pd.read_csv(dataFileNames[3], header=None)

    ax = df1.plot(x=0, y=1, linestyle=":", label="Day One", title="Plots of temperatures")
    df2.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Two")
    df3.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Three")
    df4.plot(x=0, y=1, linestyle=":", ax=ax, label="Day Four")
    ax.set_xlabel("Hour of the day (13 = 1PM)")
    ax.set_ylabel("Temperature (F)")
    return plt


epsilon = 0.005
weights = [np.random.normal(), np.random.normal()]
weightsLog = [weights]


def learn(sep_line, number_of_iterations):
    final_weights = None

    print(weights)

    trainingDataFrames = [
        pd.read_csv(dataFileNames[0], header=None),
        pd.read_csv(dataFileNames[1], header=None),
        pd.read_csv(dataFileNames[2], header=None),
    ]

    for i in range(0, number_of_iterations):
        print("iteration", i)

        # mix up the test data frame so that we learn in different ways(?)
        train_df = trainingDataFrames[i % len(trainingDataFrames)]
        # For each element in the data_frame `train_df`
        for index, row in train_df.iterrows():
            new_weights = calculate_weight_after_delta_d(sep_line, row)

            print(new_weights)

            weightsLog.append(new_weights.copy())

            sep_line[0] = new_weights[0]
            sep_line[1] = new_weights[1]

        final_weights = sep_line

    return final_weights


def build_sep_line_plot(sep_line):
    ax = None

    # Plot a vertical line at `x`
    for w in weightsLog:
        y1 = (5 * w[0]) + w[1]
        y2 = (13 * w[0]) + w[1]
        a = plt.plot([5, 13], [y1, y2])

        if ax is None:
            ax = a

            # frame1 = plt.gca()
            # frame1.axes.get_yaxis().set_visible(False)

    return plt


def calculate_weight_after_delta_d(current_weight, current_pattern, alpha=0.005, k=0.5):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1])

    output = (np.tanh(net * k) + 1) / 2
    # output = net * 1
    delta_d = alpha * (current_pattern[1] - output)

    current_pattern[0] *= delta_d
    current_pattern[1] = delta_d

    current_weight[0] += current_pattern[0]
    current_weight[1] += current_pattern[1]

    return current_weight


def main():
    myPlt = display_plot_of_temperatures()
    myPlt.savefig("images/testing_training_graph")
    myPlt.gcf().clear()

    learn(weights, 10)

    mySepLinePlot = build_sep_line_plot(weights)
    mySepLinePlot.show()

    file = open(reportFileName, "w")

    save_markdown_report(file, [
        md.h1("Project 3 Report"),
        md.h2("CMSC 409 - Artificial Intelligence"),
        md.h2("Steven Hernandez"),

        md.p("""1. There would be two input and one output for our unit.
Inputs would be the hour and a bias input while output would be the estimated
temperature at that hour of the day.
In fact, because we have weights for x (hour of the day) and a bias,
we can create the formula net = ax+b which means our unit can simply return net * 1
or the identity."""),

        md.p("""2. The activation function would be some linear function.
Or unit would not have a threshold however.
Whatever the outcome from the linear activation function is
would be the exact result from the learning unit.
If we look at the graph of temperatures for our training
(and testing) data, we can see that the values are basically
just a linear function."""),
        md.image("./images/testing_training_graph.png", "Testing training graph"),
        md.p("3. Outcome of training with days 1-3"),
    ])

    file.close()

    print("Markdown Report generated in ./report.md")
    print("Converting Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName + "`")

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName)
    print("Report created")


if __name__ == "__main__":
    main()
