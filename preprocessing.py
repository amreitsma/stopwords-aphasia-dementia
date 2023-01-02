from pylangacq import *


def preprocess_data(zipf, dtype):
    """Turns a zip file containing CHAT transcripts to a format that can be used as input for vectorization.

    zipf: zip folder with CHAT files

    dtype: patient group the data should be filtered on"""
    print("Preprocessing {}...".format(zipf))
    data = read_chat(zipf)
    words = []
    text_lens = []

    for file in data:
        if file.headers()[0]["Participants"]["PAR"]["group"] == dtype:
            wordlist = file.words(participants="PAR")
            clean_wordlist = []
            for word in wordlist:
                if "_" in word:
                    word = word.replace("_", " ")  # split compound words
                if "^" in word:
                    word = word.replace("^", "")  # join words with pauses between syllables
                clean_wordlist.append(word)
            words.append(" ".join(clean_wordlist))
            text_lens.append(len(clean_wordlist))

    # print some details about the text transcripts
    # print("Group {}\nMax word count: {}, min word count: {}, average word count: {}".format(dtype, max(text_lens),
    #                                                                                         min(text_lens),
    #                                                                                         sum(text_lens)/len(words)))
    print("Amount of transcripts: ", len(words))
    print()

    return words


def main():
    preprocess_data("Aphasia_chat.zip", "Broca")
    preprocess_data("Dementia_chat_copy.zip", "ProbableAD")


if __name__ == "__main__":
    main()
