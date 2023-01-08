# The importance of stop words in automatically distinguishing aphasia from dementia
In this repository you can find all the code used in my bachelor's thesis. Due to privacy reasons, it contains only the code and not the data used to
run the experiments. To replicate the experiments in my thesis and obtain the same scores, a collection of aphasia and dementia CHAT files are needed.
They should be named `Aphasia_chat.zip` and `Dementia_chat.zip` respectively, and must be placed in the same folder as the python scripts.

## Prerequisites

This project makes use of several external Python libraries, which are listed below:
- scikit-learn
- matplotlib
- numpy
- pylangacq
- nltk

Please ensure all of these libraries are installled before attempting to run any of the files in this project.

## Running the experiments

In the file `models.py` the `main` function contains several lines of code that can be uncommented to run the various different experiments.
The parts of the code you can uncomment have been divided into five sections:
1. Calculating the average amount of stop words per 100 words for the two patient groups and giving an overview of the top 10 most commonly used stop
words for each group.
2. Running the baseline for the main experiment
3. Running and evaluating the four models on data with and without stop words
4. Running and evaluating the four models on balanced datasets with and without stop words
5. Running and evaluating the four models on a dataset that contains only the stop words

This is in the same order as the order in which the experiments are discussed in the *Results And Discussion* chapter of my thesis.
