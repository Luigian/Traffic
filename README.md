# Traffic

## An AI to identify which traffic sign appears in a photograph

<img src="resources/images/traffic_output.png" width="1000">

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. 

This is a question answering system based on **inverse document frequency** that will perform two tasks: document retrieval and passage retrieval.

First, the system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. To find the most relevant documents, we’ll use tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. 

Then, once the top documents are found, passage retrieval will subdivide the top document(s) into passages (in this case, sentences) and score them using a combination of inverse document frequency and a query term density measure, so that the most relevant passage to the question can be determined.

## Implementation

Inside the `corpus` directory, each document is a text file containing the contents of a Wikipedia page. The AI can find sentences from these files that are relevant to a user’s query. You are welcome and encouraged to add, remove, or modify files in the corpus if you’d like to experiment with answering queries based on a different corpus of documents. Just be sure each file in the corpus is a text file ending in .txt.

At `questions.py`, in the main function, we first load the files from the corpus directory into memory (via the `load_files` function). Each of the files is then tokenized (via `tokenize`) into a list of words, which then allows us to compute inverse document frequency values for each of the words (via `compute_idfs`). The user is then prompted to enter a query. The `top_files` function identifies the files that are the best match for the query. From those files, sentences are extracted, and the `top_sentences` function identifies the sentences that are the best match for the query.

The global variable `FILE_MATCHES` specifies how many files should be matched for any given query. The global variable `SENTENCES_MATCHES` specifies how many sentences within those files should be matched for any given query. By default, each of these values is 1. The AI will find the top sentence from the top matching document as the answer to the question. 

### Loading the data

* The `load_files` function accepts the name of a directory and return a dictionary mapping the filename of each .txt file inside that directory to the file’s contents as a string.

* On macOS, the `/` character is used to separate path components, while the `\` character is used on Windows. By using `os.path.join`, this function is platform-independent, it works regardless of operating system.

* In the returned dictionary, there's one key named for each .txt file in the directory. The value associated with that key is a string (the result of reading the corresponding file).

### Extracting the words

* The `tokenize` function accepts a document (a string) as input, and return a list of all of the words in that document, in order and lowercased.

* It uses nltk’s `word_tokenize` function to perform tokenization, and filters out punctuation and stopwords (common words that are unlikely to be useful for querying). Punctuation is defined as any character in `string.punctuation`. Stopwords are defined as any word in `nltk.corpus.stopwords.words("english")`.

* If a word appears multiple times in the document, it also appears multiple times in the returned list (unless it was filtered out).

### Calculating the inverse document frecuencies

* The `compute_idfs` function accepts a dictionary of documents (a dictionary mapping names of documents to a list of words in that document) and return a new dictionary mapping words to their IDF (inverse document frequency) values. 

* The inverse document frequency of a word is defined by taking the natural logarithm of the number of documents divided by the number of documents in which the word appears.

* The returned dictionary maps every word that appears in at least one of the documents to its inverse document frequency value.

### Finding the top file matches

* The `top_files` function accepts a query (a set of words), `files` (a dictionary mapping names of files to a list of their words), and `idfs` (a dictionary mapping words to their IDF values), and return a list of the filenames of the `n` top files that match the query, ranked according to **tf-idf**.

* The tf-idf for a term is computed by multiplying the number of times the term appears in the document (**term frecuency**) by the IDF value for that term.

* Files are ranked according to the sum of tf-idf values for any word in the query that also appears in the file. Words in the query that do not appear in the file doesn't contribute to the file’s score.

* The returned list of filenames is of length `n` and is ordered with the best match first.

### Finding the top sentence matches

* The `top_sentences` function accepts a query (a set of words), `sentences` (a dictionary mapping sentences to a list of their words), and `idfs` (a dictionary mapping words to their IDF values), and return a list of the `n` top sentences that match the query, ranked according to IDF.

* Sentences are ranked according to **matching word measure**, which is the sum of IDF values for any word in the query that also appears in the sentence. Term frequency isn't taken into account here, only inverse document frequency.

* If two sentences have the same value according to the matching word measure, then sentences with a higher **query term density** are preferred. Query term density is defined as the proportion of words in the sentence that are also words in the query. For example, if a sentence has 10 words, 3 of which are in the query, then the sentence’s query term density is 0.3.

* The returned list of sentences is of length `n` and is ordered with the best match first.

## Resources
* [Language - Lecture 6 - CS50's Introduction to Artificial Intelligence with Python 2020][cs50 lecture]
* [How to Clean Text for Machine Learning with Python][clean text]
* [Python | Nested Dictionary][nested dictionary]
* [Breaking Ties: Second Sorting][second sorting]

## Installation
Inside of the `questions` directory:

* `pip3 install -r requirements.txt` | Install this project’s dependency: nltk for natural language processing.

## Usage
Inside of the `questions` directory:

* `python3 questions.py corpus` | Accepts the corpus of documents via a directory, and the query via user input.

## Credits
[*Luis Sanchez*][linkedin] 2020.

A project from the course [CS50's Introduction to Artificial Intelligence with Python 2020][cs50 ai] from HarvardX.

[cs50 lecture]: https://youtu.be/_hAVVULrZ0Q?t=4158
[clean text]: https://machinelearningmastery.com/clean-text-machine-learning-python/
[nested dictionary]: https://www.geeksforgeeks.org/python-nested-dictionary/
[second sorting]: https://runestone.academy/runestone/books/published/fopp/Sorting/SecondarySortOrder.html
[linkedin]: https://www.linkedin.com/in/luis-sanchez-13bb3b189/
[cs50 ai]: https://cs50.harvard.edu/ai/2020/

