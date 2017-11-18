import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import markdown as md
import math

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

total_errors = []
testing_errors = []


def calculate_error(data_frame, weights):
    sum = 0

    for index, row in data_frame.iterrows():
        calculated_output = row[0] * weights[0] + weights[1]
        sum += math.pow(calculated_output - row[1], 2)

    return math.sqrt(sum)


input_count = 1


def learn(number_of_iterations):
    final_weights = weights

    df1 = pd.read_csv(dataFileNames[0], header=None)
    df2 = pd.read_csv(dataFileNames[1], header=None)
    df3 = pd.read_csv(dataFileNames[2], header=None)
    test_df = pd.read_csv(dataFileNames[3], header=None)

    train_df = df1.append(df2).append(df3)

    for iter in range(0, number_of_iterations):
        print("iteration", iter)

        total_error = calculate_error(test_df, final_weights)

        testing_errors.append(total_error)

        if epsilon > total_error:
            break

        # For each element in the data_frame `train_df`
        for index, row in train_df.iterrows():
            new_weights = calculate_weight_after_delta_d(weights, row)

            for i in range(0, input_count):
                weights[i] = new_weights[i]

        final_weights = weights

        weightsLog.append(final_weights)

    return final_weights


def build_sep_line_plot():
    for w in weightsLog:
        y1 = (5 * w[0]) + w[1]
        y2 = (13 * w[0]) + w[1]
        a = plt.plot([5, 13], [y1, y2])

    return plt


def calculate_weight_after_delta_d(current_weight, current_pattern, alpha=0.0005, k=0.5):
    net = current_pattern[0] * weights[0] + weights[1]

    output = net * 1
    delta_d = alpha * (current_pattern[1] - output)

    current_pattern[0] *= delta_d
    current_pattern[1] *= delta_d

    current_weight[0] += current_pattern[0]
    current_weight[1] += current_pattern[1]

    return current_weight


def output(input):
    return input * weights[0] + weights[1]


def main():
    display_plot_of_temperatures()
    plt.savefig("images/testing_training_graph")
    plt.gcf().clear()

    learn(number_of_iterations=100)

    display_plot_of_temperatures()
    mySepLinePlot = build_sep_line_plot()
    mySepLinePlot.title("Plots of temperatures and learning unit activation function")
    plt.savefig("images/activation_line")
    plt.gcf().clear()

    plt.plot(testing_errors)
    plt.ylabel("Euclidean Distance")
    plt.xlabel("Iteration #")
    plt.title("Testing error over iterations")
    plt.savefig("images/testing_error")
    plt.gcf().clear()

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
        md.p("3. Outcome of training with days 1-3:"),
        md.p("Euclidean distance comes down from %f to %f" % (
            testing_errors[0], testing_errors[len(testing_errors) - 1])),
        md.image("./images/testing_error.png", "Testing Error"),
        md.p("resulting in an activation as so:"),
        md.image("./images/activation_line.png", "Testing Error"),
        md.p("4."),
        md.table([
            ["input", "expected output", "actual output", "Euclidean distance"],
            [5, 59.5, output(5), -59.5 + output(5)],
            [6, 64, output(6), -64 + output(6)],
            [7, 68.7, output(7), -68.7 + output(7)],
            [8, 73.65, output(8), -73.65 + output(8)],
            [9, 78.43, output(9), -78.43 + output(9)],
            [10, 82, output(10), -82 + output(10)],
            [11, 85.2, output(11), -85.2 + output(11)],
            [12, 87, output(12), -87 + output(12)],
            [13, 90.67, output(13), -90.67 + output(13)],
        ]),
        md.p("5. Learning rate was 0.0005 to keep the learning from going to quickly,"
             "while we went through 100 iterations."),
        md.p("Notice from the graph above on Euclidean distances, we reach our peak around the 20th iteration mark"),
        md.p("6. As such, after the 20th iteration, we reach a plateau of improvement with our current system."),
        md.p("7. Using a more complex network with greater than one unit would allow for more complex output"
             "which would ultimately help us with this problem."),
        md.p("Currently, we are stuck with a linear output because the single unit can only learn as such."),
    ])

    file.close()

    print("Markdown Report generated in ./report.md")
    print("Converting Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName + "`")

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName)
    print("Report created")


if __name__ == "__main__":
    main()
