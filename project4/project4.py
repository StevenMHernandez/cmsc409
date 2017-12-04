import re
import os
import math
import numpy as np
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
    split = list(map(lambda row: row[start:end], tableContents))

    def add_number_to_vector(tuple):
        vector = tuple[1]

        if tuple[0] == 0:
            vector.insert(0, "sentence #")
        else:
            vector.insert(0, str(tuple[0]))

        return vector

    return list(map(add_number_to_vector, enumerate(split)))


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


def get_closest_cluster(cluster_weights, current_pattern):
    cluster_distances = np.zeros(len(cluster_weights))

    # Euclidean distance:

    # For each cluster
    for i in range(len(cluster_weights)):
        cluster_distances[i] = 0

        # For each word
        for j in range(len(current_pattern)):
            val = pow((current_pattern[j] - cluster_weights[i][j]), 2)
            cluster_distances[i] += val

    least_distance_index = 0

    for i in range(len(cluster_distances)):
        if cluster_distances[i] < cluster_distances[least_distance_index]:
            least_distance_index = i

    return least_distance_index


def calculate_change_in_weight(current_weights, current_pattern):
    epsilon = 0.05
    return epsilon * (current_pattern - current_weights)


def learn_wta(data, cluster_count=1):
    seed = 9
    # Create randomly initialized weights for the number of expected clusters
    # Each of these weight sets with len(data[0]) number of weights
    weights = np.random.rand(cluster_count, len(data[0]))

    for i in range(1000):
        # shuffle(data)
        for pattern in data:
            index = get_closest_cluster(weights, pattern)
            weights[index] += calculate_change_in_weight(weights[index], pattern)

    return weights


def split_sentences_into_clusters(learned_weights, data):
    clustered_sentences = []

    for _ in learned_weights:
        clustered_sentences.append([])

    sentences = open("sentences/sentences.txt", 'r').read().split('\n')

    for i in range(len(data)):
        index = get_closest_cluster(learned_weights, data[i])
        clustered_sentences[index].append((i, sentences[i]))

    return clustered_sentences


def normalize_feature_vector(feature_vector):
    word_count = len(feature_vector[0])
    max_values = np.zeros(word_count)

    for i in range(word_count):
        for sentence_vector in feature_vector:
            if sentence_vector[i] > max_values[i]:
                max_values[i] = sentence_vector[i]

    for i in range(len(feature_vector)):
        for j in range(word_count):
            # feature_vector[i][j] = feature_vector[i][j] / max_values[j] + 1
            feature_vector[i][j] = 1 if feature_vector[i][j] else 0

    return feature_vector


def main():
    minimum_occurrences = 2
    encountered_words = get_encountered_words(minimum_occurrences=minimum_occurrences)
    feature_vector = create_feature_vector(minimum_occurrences=minimum_occurrences)

    table = [get_encountered_words(minimum_occurrences=minimum_occurrences)] + feature_vector

    normalized_feature_vector = normalize_feature_vector(feature_vector)

    result = learn_wta(normalized_feature_vector, cluster_count=20)

    clustered_sentences = split_sentences_into_clusters(result, normalized_feature_vector)

    clustered_sentences = list(filter(lambda x: x, clustered_sentences))

    def sentence_tuple_to_str(tuple):
        return str(tuple[0]) + ") " + tuple[1]

    clustered_sentence_strings = list(
        map(lambda cluster: list(map(sentence_tuple_to_str, cluster)), clustered_sentences))

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
        md.p("The following lists the root words with greater than " + str(minimum_occurrences) + "occurrences:"),
        md.table(split_table_rows(["Root Word", "\# of instances"], list(
            ({k: v for k, v in count_encountered_words().items() if v > minimum_occurrences}).items()), 49),
                 width=50),
        md.page_break(),
        md.p("The following 2 tables show the distribution of root words which appear at least "
             + str(minimum_occurrences) +
             " times across each "
             "document (with each row indicating one sentence) (This is the Term Document Matrix **TDM**)"),
        md.table(split_table(table, 0, math.floor(len(table[0]) / 2)), width=20),
        md.page_break(),
        md.table(split_table(table, math.floor(len(table[0]) / 2) + 1, len(table[0])), width=20),
        md.page_break(),
        md.h2("Learning"),
        md.p("We begin learning by using the 'Winner Takes All' (WTA) method which means that we begin with `n` "
             "clusters, then iterating for each document, we find the closest cluster using euclidean "
             "distance. Depending on which cluster's center (based on weight) is closest to the new document, "
             "the cluster's center's weight is changed by a value to better match the resulting pattern. Code below: "),
        md.code(function=learn_wta),
        md.code(function=get_closest_cluster),
        md.code(function=calculate_change_in_weight),
        md.page_break(),
        md.h3("Learned clusters:"),
    ])

    # Show resulting clusters
    for i in range(len(clustered_sentence_strings)):
        md.save_markdown_report(file, [
            md.p("Cluster " + str(i + 1) + ":"),
            md.li(clustered_sentence_strings[i]),
        ])

    # Show bit representation of sentence vectors
    md.save_markdown_report(file, [
        md.p("If we look at the feature vectors as a bit map showing whether a sentence has or does not have "
             "a specific word, we can begin to see the pattern of the clustering method."),
    ])

    def sentence_tuple_to_bit_string(tuple):
        return str(tuple[0]) + ") " + feature_vector_to_bit_string(feature_vector[tuple[0]])

    def feature_vector_to_bit_string(vector):
        return ''.join(map(str, vector))

    for i in range(len(clustered_sentences)):
        md.save_markdown_report(file, [
            md.p("Cluster " + str(i + 1) + ":"),
            md.li(list(map(sentence_tuple_to_bit_string, clustered_sentences[i]))),
        ])

    md.save_markdown_report(file, [
        md.p("From these bit maps, we can see that each cluster has relatively distinct columns which match"
             "across the documents of the cluster."),
        md.p("Of course, this clustering does split some groups of documents into more clusters than expected. "
             "Some clusters seem as if they could be combined to the human views. Having additional sample documents "
             "would very likely help with this issue. With these few number of documents, for example, sentence 12 "
             "'Three parking spaces in back, pets are possible with approval from the owner.' does not mention "
             "being about a 'home' or many other words which are used in other documents that truly identify it"
             "as being about a home. With more documents, we would begin to have more overlap, which could "
             "aid in finding which words provide us the most importance. Sentence 10 as well does not share enough"
             "words to be able to identify it with the provided documents."),
        md.p("Below, we can see which words these sentences share in common."),
    ])

    def sentence_tuple_to_formatted_sentence(tuple):
        formatted_sentence = []

        sentence_vector = feature_vector[tuple[0]]

        for i, v in enumerate(sentence_vector):
            if v:
                formatted_sentence.append(encountered_words[i])
        return str(tuple[0]) + ") " + ", ".join(formatted_sentence)

    for i in range(len(clustered_sentence_strings)):
        md.save_markdown_report(file, [
            md.p("Cluster " + str(i + 1) + ":"),
            md.li(list(map(sentence_tuple_to_formatted_sentence, clustered_sentences[i]))),
        ])

    md.save_markdown_report(file, [
        md.p("One problem of this method compared to a method where clusters a created as needed, was that if the "
             "random initialization of weights for the cluster were randomly generated in a bad spot, it is likely "
             "the cluster would never contain any sentences because (as the name implies) the Winner Takes All method"
             "would often find one cluster taking over most of the documents, while other clusters remained empty."),
        md.p("The solution taken here for this problem was to learn on many randomly placed clusters. Learning "
             "began with 20 clusters. From these 20 clusters however, we only end up with "
             + str(len(clustered_sentences)) +
             " clusters. Additionally, (during testing) it would some times "
             "result in clusters with only a single result, when the result would have worked better "
             "in some other already defined cluster."),
        md.p("With fewer clusters (for example 4), we occasionally ended up with good results, but often would end up"
             "with most documents stuck in one single cluster"),
        md.p("In addition to having more documents to sample, having clusters only as needed would likely improve this "
             "situation. With clusters-as-needed, clusters would only be able to contain documents within some radius "
             "of the cluster's center. If a document is found outside of this radius, then a new cluster would be "
             "formed in this place.")
    ])

    file.close()

    print("Markdown Report generated in ./report4.md")
    print("Converting Markdown file to PDF with ")
    print("`pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName + "`")

    os.system("pandoc --latex-engine=xelatex -V geometry=margin=1in -s -o FINAL_REPORT.pdf " + reportFileName)
    print("Report created")


if __name__ == "__main__":
    main()
