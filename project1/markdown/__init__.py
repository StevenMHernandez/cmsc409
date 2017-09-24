import inspect


def block_element(text):
    return text + "\n\n"


def p(text):
    return block_element(text)


def h1(text):
    return block_element("# " + text)


def h2(text):
    return block_element("## " + text)


def h3(text):
    return block_element("### " + text)


def code(text=None, function=None):
    if function is not None:
        text = implode(list(inspect.getsourcelines(function))[0])

    return block_element("```\n\n" + text + "\n```")


def image(filename, alt_text=""):
    return block_element("\ ![" + alt_text + "](" + filename + ")")


def table(contents, header=True):
    table = "| "
    for x in contents[0]:
        table += x + " | "

    table += "\n| "
    for x in contents[0]:
        table += "--- | "

    for i in range(1, len(contents)):
        table += "\n| "
        for x in contents[i]:
            table += str(x) + " | "

    return block_element(table)


def implode(array, glue=''):
    return glue.join([str(i) for i in array])
