import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import markdown as md

directory = "Project1_data/"
dataFileName = directory + "data.txt"
sepLineAFileName = directory + "sep_line_a.txt"
sepLineBFileName = directory + "sep_line_b.txt"
reportFileName = "report.md"

area = 50
alpha = 0.1


def generate_random_data():
    data_file = open(dataFileName, "w")

    for gender in range(0, 2):
        height_mean = 70 / 12 if gender == 0 else 65 / 12
        weight_mean = 200 if gender == 0 else 165

        for i in range(0, 2000):
            # generate random heights and weights in a `normalized` way
            height = np.random.normal(height_mean, 0.2)
            weight = np.random.normal(weight_mean, 20)

            data_file.write(str(height) + "," + str(weight) + "," + str(gender) + "\n")

    data_file.close()


def separate_males_and_females(data_frame, remove_y_axis=False):
    # returns: (males, females)
    return data_frame[data_frame[2] == 0], data_frame[data_frame[2] == 1]


def plot_male_and_females(data_frame, remove_y_axis=False):
    males, females = separate_males_and_females(data_frame)

    male_x = males[0]
    male_y = np.full(males[0].shape, -0.001) if remove_y_axis else males[1]

    female_x = females[0]
    female_y = np.full(females[0].shape, 0.001) if remove_y_axis else females[1]

    male_plot = plt.scatter(male_x, male_y, s=area, c=np.full(males[2].shape, 'r'), alpha=alpha)
    female_plot = plt.scatter(female_x, female_y, s=area, c=np.full(females[2].shape, 'g'), alpha=alpha)

    plt.legend((male_plot, female_plot),
               ('Male', 'Female'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    if remove_y_axis:
        plt.title("Height for Male vs Female")
        plt.xlabel("Height (ft)")
    else:
        plt.title("Weight and Height for Male vs Female")
        plt.xlabel("Height (ft)")
        plt.ylabel("Weight (lbs)")

    return plt


def build_height_plot(data_frame, sep_line):
    plot_male_and_females(data_frame, remove_y_axis=True)

    # Plot a vertical line at `x`
    x = sep_line[0][1] / sep_line[0][0]
    plt.plot([x, x], [-0.1, 0.1])

    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)

    return plt


def build_height_weight_plot(data_frame, sep_line):
    plot_male_and_females(data_frame)

    # Plot separation line
    x_weight = sep_line[0][0]
    y_weight = sep_line[0][1]
    bias = sep_line[0][2]

    # So that this separation line covers the entire of the plotted data
    # we specify the minimum x and the maximum y for the line.
    x1 = data_frame[0].min()
    x2 = data_frame[0].max()

    # formula is y_weight(y) = x_weight(x) + bias(1)
    # or y = (x_weight/a)y_weight + (bias/y_weight)
    y1 = ((x_weight * x1) / y_weight) + (bias / y_weight)
    y2 = ((x_weight * x2) / y_weight) + (bias / y_weight)

    plt.plot([x1, x2], [y1, y2])

    return plt


def eq(formula, x_range):
    return formula(x_range)


# returns 0 for female, 1 for male
def get_output_for_row(current_pattern, current_weight):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1] * current_pattern[1] +
           current_weight[2])

    return 1 if net > 0 else 0


def get_soft_output_for_row(current_pattern, current_weight, k=0.3):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1] * current_pattern[1] +
           current_weight[2])

    return np.tanh(net * 0.5)


def get_confusion_matrix(data_frame, sep_line):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in data_frame.iterrows():
        r = row[1]

        if len(sep_line) == 3:
            gender = r[2]

            if get_output_for_row(r, sep_line):
                if gender == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if gender == 0:
                    true_negative += 1
                else:
                    false_negative += 1
        else:
            height = r[0]
            weight = r[1]
            gender = r[2]
            x_weight = sep_line[0]
            bias = sep_line[1]

            # 0 <= bx - c
            net = x_weight * height - bias * 1

            if net < 0:
                if gender == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if gender == 0:
                    true_negative += 1
                else:
                    false_negative += 1

    return (true_positive,
            true_negative,
            false_positive,
            false_negative)


def save_markdown_report(file, arr):
    for block in arr:
        file.write(block)


def main():
    # Data has been generated, so we don't want to regenerate the data.
    # generate_random_data()

    df = pd.read_csv(dataFileName, header=None)
    sepLineA = pd.read_csv(sepLineAFileName, header=None)
    sepLineB = pd.read_csv(sepLineBFileName, header=None)
    #
    errorMatrix1 = get_confusion_matrix(df, sepLineA)
    errorMatrix2 = get_confusion_matrix(df, sepLineB)

    myPlt = build_height_plot(df, sepLineA)
    myPlt.savefig("images/1d")
    myPlt.gcf().clear()

    myPlt = build_height_weight_plot(df, sepLineB)
    myPlt.savefig("images/2d")
    myPlt.gcf().clear()

    file = open(reportFileName, "w")

    save_markdown_report(file, [
        md.h1("Project 1 Report"),
        md.h2("CMSC 409 - Artificial Intelligence"),
        md.h2("Steven Hernandez"),

        md.p("Fully generated data can be found in `./Project1_data/data.txt"),

        md.h3("*Scenerio 1:* using only height."),
        md.table([
            ["", "Weights"],
            ["x", sepLineA[0][0]],
            ["bias", sepLineA[0][1]]
        ]),
        md.p("Assuming the following"),
        md.image("./images/net.png"),
        md.p("Or in this situation: "),
        md.p("1 if 0 <= -a(Height) + bias, otherwise 0"),
        md.p("where *a* is some weight and *1* is male and *0* is female."),
        md.p("In this situation a=" + str(sepLineA[0][0]) + " and bias=" + str(sepLineA[0][1])),
        md.image("./images/1d.png"),
        md.table([
            ["", "Predicted Male", "Predicted Female"],
            ["Actual Male", errorMatrix1[1], errorMatrix1[2]],
            ["Actual Female", errorMatrix1[3], errorMatrix1[0]]
        ]),
        md.p("**Confusion Matrix**"),
        md.table([
            ["", ""],
            ["Error", 1 - ((errorMatrix1[1] + errorMatrix1[0]) / 4000)],
            ["Accuracy", (errorMatrix1[1] + errorMatrix1[0]) / 4000],
            ["True Positive Rate", errorMatrix1[1] / 2000],
            ["True Negative Rate", errorMatrix1[0] / 2000],
            ["False Positive Rate", errorMatrix1[3] / 2000],
            ["False Negative Rate", errorMatrix1[2] / 2000],
        ]),

        md.h3("*Scenerio 2:* heights and weights."),
        md.table([
            ["", "Weights"],
            ["x", sepLineB[0][0]],
            ["y", sepLineB[0][1]],
            ["bias", sepLineB[0][2]]
        ]),
        md.p("Assuming the following"),
        md.image("./images/net.png"),
        md.p("Or in this situation:"),
        md.p("1 if 0 <= a(Height) - b(Weight) + bias, otherwise 0"),
        md.p("where *a* and *b* are some weights and *1* is male and *0* is female."),
        md.p("In this situation a=" + str(sepLineB[0][0]) + " and b=" + str(sepLineB[0][1]) + " and bias=" + str(
            sepLineB[0][2])),
        md.image("./images/2d.png"),
        md.p("Notice, Male and Female are on slightly different levels in this graph"
             "so that one does not completely cover up the other."),
        md.p("**Confusion Matrix**"),
        md.table([
            ["", "Predicted Male", "Predicted Female"],
            ["Actual Male", errorMatrix2[1], errorMatrix2[2]],
            ["Actual Female", errorMatrix2[3], errorMatrix2[0]]
        ]),
        md.table([
            ["", ""],
            ["Error", 1 - ((errorMatrix2[1] + errorMatrix2[0]) / 4000)],
            ["Accuracy", (errorMatrix2[1] + errorMatrix2[0]) / 4000],
            ["True Positive Rate", errorMatrix2[1] / 2000],
            ["True Negative Rate", errorMatrix2[0] / 2000],
            ["False Positive Rate", errorMatrix2[3] / 2000],
            ["False Negative Rate", errorMatrix2[2] / 2000],
        ]),

        md.h3("Libraries Used"),
        md.p("matplotlib, numpy, pandas, pandoc"),

        md.h3("Selected Code Functions"),
        md.p("Functions used to generate this data and calculations."),
        md.p("The full code can be found in `./project1.py`"),
        md.code(function=generate_random_data),
        md.code(function=plot_male_and_females),
        md.code(function=plot_male_and_females),
        md.code(function=get_confusion_matrix),
    ])

    file.close()

    print("Markdown Report generated in ./report.md")
    print("Convert Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf report.md`")


if __name__ == "__main__":
    main()
