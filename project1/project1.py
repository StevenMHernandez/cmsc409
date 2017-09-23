import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import markdown as md

directory = "Project1_data/"
dataFileName = directory + "data.txt"
sepLineAFileName = directory + "sep_line_a.txt"
sepLineBFileName = directory + "sep_line_b.txt"
reportFile = "report.md"


def generate_random_data():
    file = open(dataFileName, "w")

    for gender in range(0, 2):
        heightMean = 70 / 12 if gender == 0 else 65 / 12
        weightMean = 200 if gender == 0 else 165

        for i in range(0, 2000):
            height = np.random.normal(heightMean, 0.2)
            weight = np.random.normal(weightMean, 20)
            file.write(str(height) + "," + str(weight) + "," + str(gender) + "\n")

            file.close()


def build_height_plot(data_frame, sep_line):
    area = 50

    males = data_frame[data_frame[2] == 0]
    females = data_frame[data_frame[2] == 1]

    malePlot = plt.scatter(males[0], np.full(males[0].shape, 0), s=area, c=np.full(males[2].shape, 'r'), alpha=0.5)
    femalePlot = plt.scatter(females[0], np.full(males[0].shape, 0), s=area, c=np.full(females[2].shape, 'g'),
                             alpha=0.5)

    x = sep_line[0][1] / sep_line[0][0]

    plt.plot([x, x], [-0.1, 0.1])

    plt.legend((malePlot, femalePlot),
               ('Male', 'Female'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    plt.title("Height for Male vs Female")
    plt.xlabel("Height (ft)")
    return plt


def build_height_weight_plot(data_frame, sep_line):
    area = 50

    males = data_frame[data_frame[2] == 0]
    females = data_frame[data_frame[2] == 1]

    male_plot = plt.scatter(males[0], males[1], s=area, c=np.full(males[2].shape, 'r'), alpha=0.5)
    female_plot = plt.scatter(females[0], females[1], s=area, c=np.full(females[2].shape, 'g'), alpha=0.5)

    plt.legend((male_plot, female_plot),
               ('Male', 'Female'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    # formula is yWeight(y) = xWeight(x) + bias(1)
    # or y = (xWeight/a)yWeight + (bias/yWeight)
    xWeight = sep_line[0][0]
    yWeight = sep_line[0][1]
    bias = sep_line[0][2]

    x1 = data_frame[0].min()
    x2 = data_frame[0].max()

    y1 = ((xWeight * x1) / yWeight) + (bias / yWeight)
    y2 = ((xWeight * x2) / yWeight) + (bias / yWeight)

    plt.plot([x1, x2], [y1, y2])

    plt.title("Weight and Height for Male vs Female")
    plt.xlabel("Height (ft)")
    plt.ylabel("Weight (lbs)")
    return plt


def eq(formula, x_range):
    return formula(x_range)


def get_confusion_matrix(data_frame, sep_line):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Note, this is only going to be for x,y,bias for now
    for row in data_frame.iterrows():
        r = row[1]

        if len(sep_line[0]) == 3:
            height = r[0]
            weight = r[1]
            gender = r[2]
            x_weight = sep_line[0][0]
            y_weight = sep_line[0][1]
            bias = sep_line[0][2]

            # 0 <= bx + x - ay
            if (x_weight * height) + bias - (y_weight * weight) >= 0:
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
            x_weight = sep_line[0][0]
            bias = sep_line[0][1]

            # 0 <= bx - c
            # or y = (x_weight/a)y_weight + (bias/y_weight)
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


# MAIN:

# Data has been generated, so we don't want to regenerate the data.
# generate_random_data()

df = pd.read_csv(dataFileName, header=None)
sepLineA = pd.read_csv(sepLineAFileName, header=None)
sepLineB = pd.read_csv(sepLineBFileName, header=None)
#
errorMatrix1 = get_confusion_matrix(df, sepLineA)
errorMatrix2 = get_confusion_matrix(df, sepLineB)

plt = build_height_plot(df, sepLineA)
plt.savefig("1d")
plt.gcf().clear()

plt = build_height_weight_plot(df, sepLineB)
plt.savefig("2d")
plt.gcf().clear()

file = open(reportFile, "w")

save_markdown_report(file, [
    md.h1("Project 1 Report"),
    md.h2("CMSC 409 - Artificial Intelligence"),
    md.h2("Steven Hernandez"),
    md.h3("*Scenerio 1* using only height."),
    md.table([
        ["", "Weights"],
        ["x", sepLineA[0][0]],
        ["bias", sepLineA[0][1]]
    ]),
    md.image("1d.png"),
    md.p("Assuming the following"),
    md.image("net.png"),
    md.p("Or in this situation: "),
    md.p("1 if 0 <= -a(Height) + bias, otherwise 0"),
    md.p("where *a* is some weight and *1* is male and *0* is female."),
    md.table([
        ["", "Predicted Male", "Predicted Female"],
        ["Actual Male", errorMatrix1[1], errorMatrix1[2]],
        ["Actual Female", errorMatrix1[3], errorMatrix1[0]]
    ]),
    md.h3("*Scenerio 2* heights and weights."),
    md.table([
        ["", "Weights"],
        ["x", sepLineB[0][0]],
        ["y", sepLineB[0][1]],
        ["bias", sepLineB[0][2]]
    ]),
    md.image("2d.png"),
    md.p("Assuming the following"),
    md.image("net.png"),
    md.p("Or in this situation:"),
    md.p("1 if 0 <= a(Height) - b(Weight) + bias, otherwise 0"),
    md.p("where *a* and *b* are some weights and *1* is male and *0* is female."),
    md.p("where w_i is weight and "),
    md.table([
        ["", "Predicted Male", "Predicted Female"],
        ["Actual Male", errorMatrix2[1], errorMatrix2[2]],
        ["Actual Female", errorMatrix2[3], errorMatrix2[0]]
    ]),
    md.h3("Selected Code Functions"),
    md.p("Functions used to generate this data and calculations."),
    md.code(function=get_confusion_matrix),
    md.h3("Libraries Used"),
    md.p("matplotlib, numpy, pandas")
])

file.close()
