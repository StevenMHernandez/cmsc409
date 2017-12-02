import re
import os
import math
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


def get_encountered_words(minimum_occurrences=0):
    if minimum_occurrences > 0:
        def filter_minimally_used_words(tuple):
            return tuple[1] > minimum_occurrences

        def tuple_to_value(tuple):
            return tuple[0]

        return list(map(tuple_to_value, filter(filter_minimally_used_words, count_encountered_words().items())))
    return list(count_encountered_words())


def count_encountered_words():
    reduced_sentence_list = apply_all_text_mining_techniques()

    encountered_words = {}

    for sentence in reduced_sentence_list:
        for word in sentence:
            if word in encountered_words:
                encountered_words[word] += 1
            else:
                encountered_words[word] = 1

    return encountered_words


#
# 1.
#
def create_feature_vector(minimum_occurrences=0):
    reduced_sentence_list = apply_all_text_mining_techniques()

    encountered_words = get_encountered_words(minimum_occurrences=minimum_occurrences)

    def sentence_to_feature_vector_row(sentence):
        vector = [0] * len(encountered_words)

        for word in sentence:
            # If we only want the words with `n` minimum_occurrences,
            # the word may not exist in `encountered_words`
            if encountered_words.__contains__(word):
                vector[encountered_words.index(word)] += 1

        return vector

    return list(map(sentence_to_feature_vector_row, reduced_sentence_list))


# Split table columns when table is too wide
def split_table(tableContents, start, end):
    return list(map(lambda row: row[start:end], tableContents))


# Split table rows when table takes up too many pages
# This combines multiple row into the same row to save space
def split_table_rows(title_contents, table_contents, max_rows):
    split_rows = []

    while table_contents:
        split_rows.append(table_contents[0:max_rows])
        table_contents = table_contents[max_rows:]

    column_count = len(split_rows)

    table = [title_contents * column_count]

    for i in range(len(split_rows[0])):
        new_row = []
        for group in split_rows:
            if i < len(group):
                new_row.append(group[i][0])
                new_row.append(group[i][1])
            else:
                new_row.append("")
                new_row.append("")
        table.append(new_row)

    return table


def main():
    table = [get_encountered_words(minimum_occurrences=2)] + create_feature_vector(minimum_occurrences=2)

    file = open(reportFileName, "w")
    md.save_markdown_report(file, [
        md.meta_data("Project 4 Report - CMSC 409 - Artificial Intelligence", "Steven Hernandez"),
        md.p("In total, there are " + str(len(get_encountered_words())) + " unique root words found. "),
        md.p(str(len(get_encountered_words(minimum_occurrences=2))) + " words that are encountered at least 2 times. "),
        md.p("And then only " + str(
            len(get_encountered_words(minimum_occurrences=3))) + " words that are encountered at least 3 times. "),
        md.p("These statistics are calculated based on processing the documents in the following ways:"),
        md.ol([
            "Tokenizing the sentences, which splits each sentence on the spaces to only produce a list of word/numeric "
            "tokens. This allows us to begin processing each word individually without requiring the context of the "
            "entire sentence. ",
            "Removing punctuation is required because in general, punctuation does not provide us textual context. "
            "Again, we are only looking at the similarity of sentence based on the number of occurrences of common "
            "words between the sentences. We are not trying to decifer the intent or the sentiment behind the "
            "sentence, so we do not require punctuation or even placement of words within the sentence. Just that the "
            "word exists "
            "within the sentence. ",
            "Removing numbers because numbers do not provide context about what the sentence is talking about. "
            "A book might cost $20 as would a basic microcontroller like an Arduino, but they are not related. "
            "Additional since, we removed punctuation in the previous step, we wouldn't be able to differentiate "
            "$20 from 20 miles or 20 participants, etc. ",
            "Converting upper to lower case prevents words appearing at the beginning of a sentence (with a required "
            "capital letter) from being considered a different word if it also appears in the middle of a sentence "
            "(which would be written in all lower case) ",
            "Removing stop words shrinks the total number of words that we find. More importantly though, it removes "
            "overly common words that do not provide us useful insights into the similarity of sentences. The word "
            "'the' is very likely to appear in most sentences, thus is not a useful indicator. ",
            "Stemming takes a word in past tense/future tense or plural/singular and takes the 'stem' or 'root' word. "
            "This further shrinks the overall number of words or dimensions that we must analyze. An example: run and "
            "running have the same root word, thus are very similar. ",
            "Combining stemmed words takes these common stemmed root words and combines them so that we can get a "
            "total count of the occurances of the word throughout all sentence documents."
        ], alpha=True),
        md.p("On the following page is a table listing all of these root words along with the number of occurrences of "
             "the word through the documents (the feature vector)"),
        md.page_break(),
        md.table(split_table_rows(["Root Word", "\# of instances"], list(count_encountered_words().items()), 49),
                 width=50),
        md.page_break(),
        md.p("The following 2 tables show the distribution of root words which appear at least 2 times across each "
             "document (with each row indicating one sentence) (This is the Term Document Matrix **TDM**)"),
        md.table(split_table(table, 0, math.floor(len(table[0]) / 2)), width=20),
        md.page_break(),
        md.table(split_table(table, math.floor(len(table[0]) / 2) + 1, len(table[0])), width=20),
    ])
    file.close()

    print("Markdown Report generated in ./report4.md")
    print("Converting Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName + "`")

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName)
    print("Report created")


if __name__ == "__main__":
    main()
