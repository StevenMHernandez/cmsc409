import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import markdown as md
import project1

directory = "Project1_data/"
dataFileName = directory + "data.txt"
reportFileName = "report2.md"

area = 50
alpha = 0.1
epsilon = 0.00005
ni = 1000


def plot_xy_sep_line(sep_line, data_frame, color="0.18"):
    x_weight = sep_line[0]
    y_weight = sep_line[1]
    bias = sep_line[2]

    min = data_frame[0].min()
    max = data_frame[0].max()
    mid = (min + ((max - min) / 2))

    # formula is y_weight(y) = x_weight(x) + bias(1)
    # or y = (x_weight/a)y_weight + (bias/y_weight)
    y1 = -(((x_weight * min) / y_weight) + (bias / y_weight))
    y2 = -(((x_weight * max) / y_weight) + (bias / y_weight))
    y_mid = -(((x_weight * mid) / y_weight) + (bias / y_weight))

    ax = plt.axes()
    ax.arrow(mid, y_mid, 0.05, 0.05, head_width=0.025, head_length=0.025, fc='k', ec='k', color="b")

    plt.plot([min, max], [y1, y2], color=color)

    return plt


def build_height_plot(data_frame):
    return project1.plot_male_and_females(data_frame, remove_y_axis=True)


def build_height_weight_plot(data_frame):
    return project1.plot_male_and_females(data_frame)


# Now, we normalize the data down to unipolar.
# From 0 to 1
def normalize_data_frame(dataframe):
    ndf = dataframe.copy()

    min_height = dataframe[0].min()
    max_height = dataframe[0].max()
    min_weight = dataframe[1].min()
    max_weight = dataframe[1].max()

    ndf[0] = (dataframe[0] - min_height) / (max_height - min_height)
    ndf[1] = (dataframe[1] - min_weight) / (max_weight - min_weight)
    return ndf


def calculate_weight_after_delta_d(current_weight, current_pattern, hard_activation=True, alpha=alpha, k=0.5):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1] * current_pattern[1] +
           current_weight[2])

    if hard_activation:
        output = 1 if net > 0 else 0
        delta_d = alpha * (current_pattern[2] - output)
    else:
        output = (np.tanh(net * k) + 1) / 2
        delta_d = alpha * (current_pattern[2] - output)

    current_pattern[0] *= delta_d
    current_pattern[1] *= delta_d
    current_pattern[2] = delta_d

    current_weight[0] += current_pattern[0]
    current_weight[1] += current_pattern[1]
    current_weight[2] += current_pattern[2]

    return current_weight


errors = []
total_errors = []
weights = [[], [], []]


def calculate_error(data_frame, sep_line):
    error_matrix = project1.get_confusion_matrix(data_frame, sep_line)
    return 1 - ((error_matrix[1] + error_matrix[0]) / (data_frame[0].count()))


def learn(train_df, test_df, sep_line, number_of_iterations, hard):
    final_sep_line = None

    for i in range(0, number_of_iterations):
        print("iteration", i)

        total_error = calculate_error(test_df, sep_line)
        total_errors.append(total_error)

        plot_xy_sep_line(sep_line, test_df, color=str(i / number_of_iterations))

        err = calculate_error(train_df, sep_line)
        errors.append(err)

        weights[0].append(sep_line[0])
        weights[1].append(sep_line[1])
        weights[2].append(sep_line[2])

        if epsilon > total_error:
            break

        # mix up the test data frame so that we learn in different ways(?)
        train_df = train_df.sample(frac=1)
        # For each element in the data_frame `train_df`
        for index, row in train_df.iterrows():
            new_weights = calculate_weight_after_delta_d(sep_line, row, hard_activation=hard)

            sep_line[0] = new_weights[0]
            sep_line[1] = new_weights[1]
            sep_line[2] = new_weights[2]

        final_sep_line = sep_line

    return final_sep_line


def build_graphs(full_data_frame, folder_destination, original_sep_line, final_sep_line):
    plt.figure(2)
    build_height_weight_plot(full_data_frame)
    plot_xy_sep_line(original_sep_line, full_data_frame, color="g")
    plot_xy_sep_line(final_sep_line, full_data_frame, color="b")

    plt.figure(1)
    plt.axis((0, 1, 0, 1))
    plt.savefig(folder_destination + "all_sep_lines")
    plt.gcf().clear()
    plt.figure(2)
    plt.axis((0, 1, 0, 1))
    plt.savefig(folder_destination + "start_end_lines")
    plt.gcf().clear()

    plt.figure(3)
    plt.plot(np.arange(len(errors)), errors, color="b")
    plt.plot(np.arange(len(total_errors)), total_errors, color="gray")
    plt.title("% error over iteration")
    plt.xlabel("Iteration number")
    plt.ylabel("% Error")
    plt.savefig(folder_destination + "error")
    plt.gcf().clear()

    plt.figure(4)
    plt.plot(np.arange(len(weights[0])), weights[0], color='r')
    plt.plot(np.arange(len(weights[1])), weights[1], color='g')
    plt.plot(np.arange(len(weights[2])), weights[2], color='b')
    plt.title("Weights over iterations")
    plt.xlabel("Iteration number")
    plt.ylabel("Weights")
    plt.savefig(folder_destination + "weights")
    plt.gcf().clear()


def build_report_file(filename, arr):
    file = open(filename, 'w')

    for item in arr:
        file.write("%s\n" % item)


def build_report_file_from_weights_arrays(filename, arr):
    file = open(filename, 'w')

    for i in range(0, len(arr[0])):
        file.write("%s,%s,%s\n" % (arr[0][i], arr[1][i], arr[2][i]))


def build_all_report_files(filepath):
    build_report_file(filepath + "errors.txt", errors)
    build_report_file(filepath + "total_errors.txt", total_errors)
    build_report_file_from_weights_arrays(filepath + "weights.txt", weights)


def reset_global_values():
    errors.clear()
    total_errors.clear()
    weights[0].clear()
    weights[1].clear()
    weights[2].clear()
    for i in range(1, 5):
        plt.figure(i)
        plt.gcf().clear()


def render(plt, hard=True, sampleFraction=0.25):
    df = pd.read_csv(dataFileName, header=None)

    df = normalize_data_frame(df)

    # smaller amount of random items
    train_df = df.sample(frac=sampleFraction)

    test_df = df[~df.isin(train_df)]

    plt.figure(1)
    plt = build_height_weight_plot(df)
    plt.figure(2)
    plt = build_height_weight_plot(df)

    rand_x = 0.1
    sep_line = [random.uniform(-rand_x, rand_x), random.uniform(-rand_x, rand_x), random.uniform(-rand_x, rand_x)]
    original_sep_line = sep_line

    plt.figure(1)
    final_sep_line = learn(train_df, test_df, sep_line, ni, hard)

    plt.figure(1)
    build_height_weight_plot(df)
    plot_xy_sep_line(original_sep_line, df, color="g")
    plot_xy_sep_line(sep_line, df, color="b")

    # Store graph images
    folder_destination = "images/project2/"
    folder_destination += "hard/" if hard else "soft/"
    folder_destination += str(int(100 * sampleFraction)) + "_"
    build_graphs(df, folder_destination, original_sep_line, final_sep_line)

    # Save graphed values to a reports file so that we don't lose data
    folder_destination = "data/project2/"
    folder_destination += "hard/" if hard else "soft/"
    folder_destination += str(int(100 * sampleFraction)) + "_"
    build_all_report_files(folder_destination)

    # reset global values
    reset_global_values()


def build_report():
    file = open(reportFileName, "w")
    project1.save_markdown_report(file, [
        md.h1("Project 2 Report"),
        md.h2("CMSC 409 - Artificial Intelligence"),
        md.h2("Steven Hernandez"),
    ])

    for activation_type in ("hard", "soft"):
        for sample_size in ("25", "50", "75"):
            project1.save_markdown_report(file, [
                md.h3(activation_type + " activation with a sample size of " + sample_size + "%"),
                md.images([
                    ["./images/project2/" + activation_type + "/" + sample_size + "_start_end_lines.png", ""],
                    ["./images/project2/" + activation_type + "/" + sample_size + "_all_sep_lines.png", ""],
                ]),
                md.images([
                    ["./images/project2/" + activation_type + "/" + sample_size + "_error.png", "errors"],
                    ["./images/project2/" + activation_type + "/" + sample_size + "_weights.png", "weights"],
                ]),
            ])

    file.close()

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT_2.pdf report2.md")
    print("Report created")


def main(plt):
    render(plt, hard=True, sampleFraction=0.25)
    render(plt, hard=True, sampleFraction=0.5)
    render(plt, hard=True, sampleFraction=0.75)

    render(plt, hard=False, sampleFraction=0.25)
    render(plt, hard=False, sampleFraction=0.5)
    render(plt, hard=False, sampleFraction=0.75)

    build_report()


if __name__ == "__main__":
    main(plt)
