import os
import sys
import re
import copy
import math


class TwitterPost:
    def __init__(self, id, username, language, tweet):
        self.id = id
        self.username = username
        self.language = language
        self.tweet = tweet


# Serves as an interface to call appropriate models
def executeNaiveBayesClassification(V, n, gamma, training, testing):

    twitter_posts = processTrainingData(training)

    if n == 1:
        executeUnigram(V, gamma, n, twitter_posts, testing)

    if n == 2:
        # implement executeBigram()
        pass

    if n == 3:
        # implement executeTrigram()
        pass

    pass


# Character Unigram Models
def executeUnigram(V, gamma, n, training, testing):

    testingData = processTestinggData(testing)

    # language frequencies denoted by _f
    eu_f, ca_f, gl_f, es_f, en_f, pt_f = buildUnigramModelByVocabulary(V, training)

    for twitterPost in testingData:
        language, probability = detectTweetLanguage(
            V, gamma, twitterPost, eu_f, ca_f, gl_f, es_f, en_f, pt_f
        )
        writeToTraceFile(twitterPost, language, probability, V, n, gamma)


# fetch and stores training data in list
def processTrainingData(training):
    trainingData = []
    with open(training, "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0:
                trainingData.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )
        return trainingData


# fetch and stores test data in list
def processTestinggData(testing):
    testingData = []
    with open(testing, "r", encoding="utf8") as f:
        for line in f:
            component = line.split(None, 3)
            if len(component) != 0:
                testingData.append(
                    # id, username, language, tweet
                    TwitterPost(component[0], component[1], component[2], component[3])
                )
    return testingData


# Returns the language with the highest probability for each test tweet
def detectTweetLanguage(V, gamma, twitterPost, eu_f, ca_f, gl_f, es_f, en_f, pt_f):

    # language frequencies denoted by _f
    eu_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, eu_f, V)
    ca_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, ca_f, V)
    gl_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, gl_f, V)
    es_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, es_f, V)
    en_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, en_f, V)
    pt_f = addMissingFrequenciesFromTestTweet(twitterPost.tweet, pt_f, V)

    # Language conditional probabilities denoted by _p
    eu_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, eu_f))
    ca_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, ca_f))
    gl_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, gl_f))
    es_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, es_f))
    en_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, en_f))
    pt_p = convertFrequenciesIntoProbabilities(addSmoothing(gamma, pt_f))

    probabilities = {}
    probabilities["eu"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, eu_p, V, gamma
    )
    probabilities["ca"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, ca_p, V, gamma
    )
    probabilities["gl"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, gl_p, V, gamma
    )
    probabilities["es"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, es_p, V, gamma
    )
    probabilities["en"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, en_p, V, gamma
    )
    probabilities["pt"] = calculateProbabilityForTestTweet(
        twitterPost.tweet, pt_p, V, gamma
    )

    language = max(probabilities)
    probability = probabilities[language]

    return language, probability


# Adds the missing frequencies found in the test tweets but not in the training tweets
def addMissingFrequenciesFromTestTweet(tweet, language, V):
    if V == 0:
        pattern = re.compile("[a-z]")
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            letter = letter.lower()
            nextLetter = tweet[i + 1]
            nextLetter = nextLetter.lower()
            if pattern.match(letter) is None or pattern.match(nextLetter) is None:
                continue
            if letter not in language.keys():
                language[letter] = {}
                language[letter][nextLetter] = 0
            if letter in language.keys() and nextLetter not in language[letter].keys():
                language[letter][nextLetter] = 0
    if V == 1:
        pattern = re.compile("[A-Za-z]")
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            nextLetter = tweet[i + 1]
            if pattern.match(letter) is None or pattern.match(nextLetter) is None:
                continue
            if letter not in language.keys():
                language[letter] = {}
                language[letter][nextLetter] = 0
            if letter in language.keys() and nextLetter not in language[letter].keys():
                language[letter][nextLetter] = 0
    if V == 2:
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            nextLetter = tweet[i + 1]
            if letter.isalpha() is False or nextLetter.isalpha() is False:
                continue
            if letter not in language.keys():
                language[letter] = {}
                language[letter][nextLetter] = 0
            if letter in language.keys() and nextLetter not in language[letter].keys():
                language[letter][nextLetter] = 0
    return language


# Iterates through test tweet letters and returns the probability of whether it belongs to a given language
def calculateProbabilityForTestTweet(tweet, language, V, gamma):

    language_probability = 0

    if V == 0:
        pattern = re.compile("[a-z]")
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            letter = letter.lower()
            nextLetter = tweet[i + 1]
            nextLetter = nextLetter.lower()
            if pattern.match(letter) is None or pattern.match(nextLetter) is None:
                continue
            if language[letter][nextLetter] == 0:
                language_probability += -math.inf
            else:
                language_probability += math.log(language[letter][nextLetter])
    elif V == 1:
        pattern = re.compile("[A-Za-z]")
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            nextLetter = tweet[i + 1]
            if pattern.match(letter) is None or pattern.match(nextLetter) is None:
                continue
            if language[letter][nextLetter] == 0:
                language_probability += -math.inf
            else:
                language_probability += math.log(language[letter][nextLetter])
    elif V == 2:
        for i in range(len(tweet) - 2):
            letter = tweet[i]
            nextLetter = tweet[i + 1]
            if letter.isalpha() is False or nextLetter.isalpha() is False:
                continue
            if language[letter][nextLetter] == 0:
                language_probability += -math.inf
            else:
                language_probability += math.log(language[letter][nextLetter])

    return language_probability


# Migrates frequencies into conditional probabilities in new 2-D dictionaries
def convertFrequenciesIntoProbabilities(frequencies):
    probabilities = copy.deepcopy(frequencies)
    row_total = 0
    for letter in frequencies.keys():
        for next_letter in frequencies[letter].keys():
            row_total += frequencies[letter][next_letter]
        for next_letter in frequencies[letter].keys():
            probabilities[letter][next_letter] = (
                probabilities[letter][next_letter] / row_total
            )
    return probabilities


# Adds gamma to all cells of 2-D dictionaries
def addSmoothing(gamma, frequencies):
    if gamma != 0 and gamma <= 1:
        for letter in frequencies.keys():
            for next_letter in frequencies[letter].keys():
                frequencies[letter][next_letter] += gamma
    return frequencies


# builds a 2-D dictionary for each language
def buildUnigramModelByVocabulary(V, twitter_posts):

    eu_letters = {}  # Basque
    ca_letters = {}  # Catalan
    gl_letters = {}  # Galician
    es_letters = {}  # Spanish
    en_letters = {}  # English
    pt_letters = {}  # Portuguese

    if V == 0:
        pattern = re.compile("[a-z]")
    elif V == 1:
        pattern = re.compile("[A-Za-z]")
    elif V == 2:
        pattern = None

    for twitter_post in twitter_posts:
        if twitter_post.language == "eu":
            buildUnigramLanguageModel(eu_letters, twitter_post, V, pattern)
        if twitter_post.language == "ca":
            buildUnigramLanguageModel(ca_letters, twitter_post, V, pattern)
        if twitter_post.language == "gl":
            buildUnigramLanguageModel(gl_letters, twitter_post, V, pattern)
        if twitter_post.language == "es":
            buildUnigramLanguageModel(es_letters, twitter_post, V, pattern)
        if twitter_post.language == "en":
            buildUnigramLanguageModel(en_letters, twitter_post, V, pattern)
        if twitter_post.language == "pt":
            buildUnigramLanguageModel(pt_letters, twitter_post, V, pattern)

    return (eu_letters, ca_letters, gl_letters, es_letters, en_letters, pt_letters)


# builds a 2-D dictionary for a given language
def buildUnigramLanguageModel(language_letters, twitter_post, V, pattern):
    if V == 0:
        tweet = twitter_post.tweet
        for i in range(len(tweet) - 1):
            letter = tweet[i]
            letter = letter.lower()
            if pattern.match(letter) is None:
                continue
            if letter not in language_letters.keys():
                language_letters[letter] = {}
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1].lower()
                    if pattern.match(next_letter) is None:
                        continue
                    language_letters[letter][next_letter] = 1
            else:
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1].lower()
                    if pattern.match(next_letter) is None:
                        continue
                    if next_letter not in language_letters[letter].keys():
                        language_letters[letter][next_letter] = 1
                    else:
                        language_letters[letter][next_letter] += 1

    if V == 1:
        tweet = twitter_post.tweet
        for i in range(len(tweet) - 1):
            letter = tweet[i]
            if pattern.match(letter) is None:
                continue
            if letter not in language_letters.keys():
                language_letters[letter] = {}
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1]
                    if pattern.match(next_letter) is None:
                        continue
                    language_letters[letter][next_letter] = 1
            else:
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1]
                    if pattern.match(next_letter) is None:
                        continue
                    if next_letter not in language_letters[letter].keys():
                        language_letters[letter][next_letter] = 1
                    else:
                        language_letters[letter][next_letter] += 1
    if V == 2:
        tweet = twitter_post.tweet
        for i in range(len(tweet) - 1):
            letter = tweet[i]
            if letter.isalpha() is False:
                continue
            if letter not in language_letters.keys():
                language_letters[letter] = {}
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1]
                    if next_letter.isalpha() is False:
                        continue
                    language_letters[letter][next_letter] = 1
            else:
                if i <= len(tweet) - 2:
                    next_letter = tweet[i + 1]
                    if next_letter.isalpha() is False:
                        continue
                    if next_letter not in language_letters[letter].keys():
                        language_letters[letter][next_letter] = 1
                    else:
                        language_letters[letter][next_letter] += 1


# writes final answers in trace file
def writeToTraceFile(twitterPost, language, probability, V, n, gamma):

    if language == twitterPost.language:
        label = "correct"
    else:
        label = "wrong"

    with open(
        "trace_" + str(V) + "_" + str(n) + "_" + str(gamma) + ".txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            twitterPost.id
            + "  "
            + language
            + "  "
            + "{:.2e}".format(probability)
            + "  "
            + twitterPost.language
            + "  "
            + label
            + "\n"
        )


# following borrowed function is used to circumvent encoding errors when printing dictionaries with isalpha allowed characters
# https://stackoverflow.com/questions/14630288/unicodeencodeerror-charmap-codec-cant-encode-character-maps-to-undefined
def uprint(*objects, sep=" ", end="\n", file=sys.stdout):
    enc = file.encoding
    if enc == "UTF-8":
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors="backslashreplace").decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)


def main():
    V = 0
    n = 1
    gamma = 0.1
    training = os.path.join(sys.path[0], "training-tweets.txt")
    testing = os.path.join(sys.path[0], "test-tweets-given.txt")
    executeNaiveBayesClassification(V, n, gamma, training, testing)


if __name__ == "__main__":
    main()
