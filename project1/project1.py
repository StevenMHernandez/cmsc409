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

    x = sep_line[0][0] + sep_line[0][1]

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

    # formula is ay = bx + c(1)
    # or y = (b/a)x + (c/a)
    xWeight = sep_line[0][0]
    yWeight = sep_line[0][1]
    bias = sep_line[0][2]

    x1 = data_frame[0].min()
    x2 = data_frame[0].max()

    y1 = ((xWeight * x1) / yWeight) + (bias / yWeight)
    y2 = ((xWeight * x2) / yWeight) + (bias / yWeight)

    print(yWeight, xWeight, bias)
    print(x1, x2, y1, y2)

    plt.plot([x1, x2], [y1, y2])

    # Try again . . .
    # r = range(math.floor(x1) - 1, math.ceil(x2) + 1)
    # print(r)
    # x = np.array(r)
    # ^ use x as range variable
    # y = eq(lambda lambda_x: ((xWeight * lambda_x) + bias) / yWeight, x)
    # ^          ^call the lambda expression with x
    # | use y as function result
    # plt.plot(x, y)

    plt.title("Weight and Height for Male vs Female")
    plt.xlabel("Height (ft)")
    plt.ylabel("Weight (lbs)")
    return plt


def eq(formula, x_range):
    return formula(x_range)


def get_confusion_matrix(data_frame, sep_line):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    # Note, this is only going to be for x,y,bias for now
    for row in data_frame.iterrows():
        r = row[1]
        height = r[0]
        weight = r[1]
        gender = r[2]

        xWeight = sep_line[0][0]
        yWeight = sep_line[0][1]
        bias = sep_line[0][2]

        # print("y = " + str(xWeight / yWeight) + "x" + " + " + str(bias / yWeight))

        # 0 = bx + x - ay
        if (xWeight * height) + bias - (yWeight * weight) >= 0:
            if gender == 1:
                truePositive += 1
            else:
                falsePositive += 1
        else:
            if gender == 0:
                trueNegative += 1
            else:
                falseNegative += 1

    return (truePositive,
            trueNegative,
            falsePositive,
            falseNegative)


# def graph(formula, x_range):

# MAIN:


# Data has been generated, so we don't want to regenerate the data.
generate_random_data()

df = pd.read_csv(dataFileName, header=None)
sepLineA = pd.read_csv(sepLineAFileName, header=None)
sepLineB = pd.read_csv(sepLineBFileName, header=None)
#
# # Load just 100 of each category.
# # Note that Males are 0 - 999 and Females are 1000 - 1999
# # render_height_graph(df, sepLineA)
#
# m = get_confusion_matrix(df[1900:2100], sepLineB)
#
m = get_confusion_matrix(df, sepLineB)

print("True Positive: ", m[0])
print("True Negative: ", m[1])
print("False Positive:", m[2])
print("False Negative:", m[3])

# print(m)
#
# # Load just 100 of each category.
# # Note that Males are 0 - 999 and Females are 1000 - 1999
# render_height_weight_graph(df[1900:2100], sepLineB)
plt = build_height_weight_plot(df, sepLineB)
plt.savefig("test")

file = open(reportFile, "w")
file.write(md.h1("Project 1 Report"))
file.write(md.h2("CMSC 409 - Artificial Intelligence"))
file.write(md.h2("Steven Hernandez"))
file.write(md.image("test.png"))
file.write(md.code(function=get_confusion_matrix))
file.write(md.table([
    ["", "Predicted Male", "Predicted Female"],
    ["Actual Male", m[1], m[2], "test"],
    ["Actual Female", m[3], m[0]]
]))
file.close()
