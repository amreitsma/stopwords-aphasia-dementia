from pylangacq import *
import re
from nltk.corpus import stopwords
from statistics import mean, stdev, median
from collections import Counter


def average_stopwds(zipf, dtype, printavg=False):
    """
    For a given zip file and patient group, returns a list of stop words extracted from the documents to be used
    as features.

    Set printavg to True to

    - Print the average amount of stop words over all transcripts

    - Print the standard deviation of the average amount of stop words

    - Print a list of top ten most frequent stop words in the documents"""
    print("Processing...")
    data = read_chat(zipf)
    p = re.compile(r"(?u)(?!\bx\b)(\b\w+\b)")
    avg_stopwds = []
    feats = []
    stop_list = []

    for file in data:
        if file.headers()[0]["Participants"]["PAR"]["group"] == dtype:
            wordlist = file.words(participants="PAR")
            file_len = 0
            stopwdcount = 0
            stopwds = []
            clean_wordlist = []
            for word in wordlist:
                word = word.lower()
                if "_" in word:
                    word = word.replace("_", " ")  # split compound words
                if "^" in word:
                    word = word.replace("^", "")  # join words with pauses between syllables

                matches = re.findall(p, word)
                file_len += len(matches)
                clean_wordlist += matches
                if len(matches) != 0:
                    for i in matches:
                        if i in stopwords.words("english"):
                            stopwdcount += 1
                            stopwds.append(i)
                            stop_list.append(i)
            feats.append(" ".join(stopwds))
            avg_stopwds.append(stopwdcount/file_len * 100)
    if printavg:
        printaverages(dtype, avg_stopwds, stop_list)
    return feats


def printaverages(dtype, avg_stopwds, stop_list):
    """Print facts about stop word use in the given patient group"""
    print("The median amount of stop words per 100 words for group {} is {}.".format(dtype, median(avg_stopwds)))
    print("The average amount of stop words per 100 words for group {} is {}.".format(dtype, mean(avg_stopwds)))
    print("The standard deviation of stop words per 100 words for group {} is {}.".format(dtype, stdev(avg_stopwds)))
    print()
    print("The ten most commonly used stop words in group {}:".format(dtype))
    print("WORD\tOCCURRENCES IN ALL DOCS OF GROUP\n------------")
    for word, count in Counter(stop_list).most_common()[:10]:
        print(word + "\t" + str(count))
    print()


def main():
    average_stopwds("Aphasia_chat.zip", "Broca", True)
    average_stopwds("Dementia_chat.zip", "ProbableAD", True)


if __name__ == "__main__":
    main()
