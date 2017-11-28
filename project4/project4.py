import re
import os
import examplecode.Porter_Stemmer_Python as PorterStemmer
import markdown as md

reportFileName = "report4.md"


def load_stop_words():
    return open("sentences/stop_words.txt", 'r').read().split('\n')


#
# 1.B & 1.C & 1.D
#
def load_sentences():
    return re.sub(r"[^A-z \n]", "", open("sentences/sentences.txt", 'r').read().lower()).split('\n')


#
# 1.A
#
def tokenize_sentences(sentence_list):
    return list(map(lambda sentence: sentence.split(), sentence_list))


#
# 1.E
#
def remove_stop_words_from_all_sentences(sentence_list):
    stop_words = load_stop_words()

    def remove_stop_words_from_one_sentence(sentence):
        return list(filter(lambda word: not stop_words.__contains__(word), sentence))

    return list(map(remove_stop_words_from_one_sentence, sentence_list))


#
# 1.F
#
def perform_stemming(sentence_list):
    stemmer = PorterStemmer.PorterStemmer()

    def remove_stop_words_from_one_sentence(sentence):
        return list(map(lambda word: stemmer.stem(word, 0, len(word) - 1), sentence))

    return list(map(remove_stop_words_from_one_sentence, sentence_list))


#
# 1.
#
def apply_all_text_mining_techniques():
    return perform_stemming(remove_stop_words_from_all_sentences(tokenize_sentences(load_sentences())))


def get_encountered_words():
    reduced_sentence_list = apply_all_text_mining_techniques()

    encountered_words = []

    for sentence in reduced_sentence_list:
        for word in sentence:
            if not encountered_words.__contains__(word):
                encountered_words.append(word)

    return encountered_words


#
# 1.
#
def create_feature_vector():
    reduced_sentence_list = apply_all_text_mining_techniques()

    encountered_words = get_encountered_words()

    def sentence_to_feature_vector_row(sentence):
        vector = [0] * len(encountered_words)

        for word in sentence:
            vector[encountered_words.index(word)] += 1

        return vector

    return list(map(sentence_to_feature_vector_row, reduced_sentence_list))


def main():
    print(apply_all_text_mining_techniques())

    table = [get_encountered_words]

    vecs = create_feature_vector()

    for vector in vecs:
        print(vector)

    file = open(reportFileName, "w")
    md.save_markdown_report(file, [
        md.meta_data("Project 2 Report - CMSC 409 - Artificial Intelligence", "Steven Hernandez"),
        md.table([get_encountered_words()] + vecs, width=20),
    ])
    file.close()

    print("Markdown Report generated in ./report4.md")
    print("Converting Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName + "`")

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName)
    print("Report created")


if __name__ == "__main__":
    main()
