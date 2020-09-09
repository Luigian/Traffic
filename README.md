# Traffic

## An AI to identify which traffic sign appears in a photograph

<img src="resources/images/traffic_output.png" width="1000">

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, I used TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, I needed a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, I used the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Implementation

The `gtsrb` directory contains 43 subdirectories numbered `0` through `42`. Each numbered subdirectory represents a different category (a different type of road sign). Within each traffic sign’s directory is a collection of images of that type of traffic sign.

In `traffic.py` the `main` function accept as command-line arguments a directory containing the data and (optionally) a filename to which to save the trained model. The data and corresponding labels are then loaded from the data directory (via the `load_data` function) and split into training and testing sets. After that, the `get_model` function is called to obtain a compiled neural network that is then fitted on the training data. The model is then evaluated on the testing data. Finally, if a model filename was provided, the trained model is saved to disk.

### Loading the data

* The `load_data` function accepts as an argument `data_dir`, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.

* We may assume that `data_dir` will contain one directory named after each category, numbered `0` through `NUM_CATEGORIES` - 1. Inside each category directory will be some number of image files.

* The OpenCV-Python module (`cv2`) is used to read each image as a `numpy.ndarray` (a numpy multidimensional array). To pass these images into a neural network, the images need to be the same size, so each image is being resized to have width `IMG_WIDTH` and height `IMG_HEIGHT`.

* The function return a tuple `(images, labels)`. `images` is a list of all of the images in the data set, where each image is represented as a `numpy.ndarray` of the appropriate size. `labels` is a list of integers, representing the category number for each of the corresponding images in the `images` list.

* On macOS, the `/` character is used to separate path components, while the `\` character is used on Windows. By using `os.path.join`, this function is platform-independent, it works regardless of operating system.

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

