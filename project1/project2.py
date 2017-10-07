import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from md2pdf.core import md2pdf
import random
import markdown as md
import project1

directory = "Project1_data/"
dataFileName = directory + "data.txt"
sepLineAFileName = directory + "sep_line_a.txt"
sepLineBFileName = directory + "sep_line_b.txt"
reportFileName = "report.md"

area = 50
alpha = 0.1


def plot_xy_sep_line(sep_line, data_frame, color="0.18"):
    x_weight = sep_line[0]
    y_weight = sep_line[1]
    bias = sep_line[2]

    min = data_frame[0].min()
    max = data_frame[0].max()

    # min = -1
    # max = 3

    # formula is y_weight(y) = x_weight(x) + bias(1)
    # or y = (x_weight/a)y_weight + (bias/y_weight)
    y1 = (((x_weight * min) / y_weight) + (bias / y_weight))
    y2 = (((x_weight * max) / y_weight) + (bias / y_weight))

    plt.plot([min, max], [y1, y2], color=color)

    return plt


def build_height_plot(data_frame):
    project1.plot_male_and_females(data_frame, remove_y_axis=True)

    return plt


def build_height_weight_plot(data_frame):
    project1.plot_male_and_females(data_frame)

    return plt


# MAIN:

df = pd.read_csv(dataFileName, header=None)


# Now, we normalize the data down to unipolar.
# From 0 to 1
min_height = df[0].min()
max_height = df[0].max()
min_weight = df[1].min()
max_weight = df[1].max()
df[0] = (df[0] - min_height) / (max_height - min_height)
df[1] = (df[1] - min_weight) / (max_weight - min_weight)


# smaller amount of random items
df = df.sample(frac=1)
df = df[0:100]

# print("d", df.iloc[0])

sep_lines_xy = []

plt = build_height_weight_plot(df)

# sep_line = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
sep_line = [-0.10005382279023051, 0.14711225825788532, 0.09040375150804325]
original_sep_line = sep_line

epsilon = 0.00005
ni = 5
alpha = 0.003

# delta(w) = alpha * x(desired-output)

errors = []

# Repeat this `n` times
n = ni

for i in range(0, n):
    # For each element in the data_frame `df`
    j = 0
    for index, row in df.iterrows():
        number_of_rows = df[0].count()
        plot_xy_sep_line(sep_line, df, color=str(((number_of_rows * i) + j) / (n * number_of_rows)))
        # do something to the sep_line, then print
        height = row[0]
        weight = row[1]
        gender = 1 if row[2] else 0

        output = 1 if project1.get_output_for_row(row, sep_line) else 0
        desired = gender

        # if output != desired:
        delta_w = alpha * (desired - output)  # *x
        sep_line[0] += delta_w * height
        sep_line[1] += delta_w * weight
        sep_line[2] += delta_w * gender

        # print bad good stuff
        errorMatrix2 = project1.get_confusion_matrix(df, sep_line)
        err = 1 - ((errorMatrix2[1] + errorMatrix2[0]) / (df.size / 3))
        print(i, err, "(", desired, "==?", output, ")", delta_w, sep_line, errorMatrix2)
        errors.append(err)
        j += 1
    err = 1 - ((errorMatrix2[1] + errorMatrix2[0]) / (df.size / 3))
    if epsilon > err:
        break

print("bme", sep_line, original_sep_line)
plot_xy_sep_line(sep_line, df, color="r")
plot_xy_sep_line(original_sep_line, df, color="b")
print(sep_line[0], original_sep_line[0])


# plot_xy_sep_line(sep_line, df, color="r")

plt.show()
# plt.savefig("images/2d")
plt.gcf().clear()


plt.plot(np.arange(len(errors)), errors)
plt.title("% error over iteration")
plt.xlabel("Iteration number")
plt.ylabel("% Error")
plt.show()
