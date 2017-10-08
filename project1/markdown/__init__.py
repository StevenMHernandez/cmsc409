import inspect


def meta_data(title, author):
    string = """---
title: "{0}"
author: "{1}"
header-includes:
- \\usepackage{{fancyhdr}}
- \\pagestyle{{fancy}}
- \\fancyhead[CO,CE]{{{0} - {1}}}
---
    """

    return block_element(string.format(title, author))


def save_markdown_report(file, arr):
    for block in arr:
        file.write(block)


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
    return images([filename, alt_text])


def images(images=[]):
    str = ""

    for i in images:
        str += "\ ![" + i[1] + "](" + i[0] + "){#id .class width=300 height=200} "

    return block_element(str)


def table(contents, header=True, width=3):
    table = "| "
    for x in contents[0]:
        table += x + " | "

    table += "\n| "
    for x in contents[0]:
        table += ("-" * width) + " | "

    for i in range(1, len(contents)):
        table += "\n| "
        for x in contents[i]:
            table += str(x) + " | "

    return block_element(table)


def implode(array, glue=''):
    return glue.join([str(i) for i in array])


def page_break():
    return block_element("\pagebreak")
