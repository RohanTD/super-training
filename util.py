import os
import openai

openai.api_key = "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"


def extract_keywords(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Extract medical keywords from this text:\n\n" + text,
        temperature=0,
        max_tokens=2046,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )
    addition = response.choices[0].text
    if addition.count("\n") > 0:
        addition = addition[addition.index("\n") :]
        addition = addition.replace("\n", "")
        if addition.count(":") > 0:
            addition = addition[addition.index(":") :]
        return addition
    else:
        return ""


def extract_keywords2(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Extract medical keywords from this text:\n\n" + text,
        temperature=0,
        max_tokens=2046,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0,
    )
    addition = response.choices[0].text
    if addition.count("\n") > 0:
        addition = addition[addition.index("\n") :]
        addition = addition.replace("\n", "")
        return addition
    else:
        return ""


def get_embedding(text, model="text-similarity-davinci-001"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


import re
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer

nltk.download("stopwords")
from nltk.corpus import stopwords

# needed for nltk.pos_tag function nltk.download(’averaged_perceptron_tagger’)
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer


def process_text(text):
    word_punct_token = WordPunctTokenizer().tokenize(text)
    clean_token = []
    for token in word_punct_token:
        token = token.lower()
        # remove any value that are not alphabetical
        new_token = re.sub(r"[^a-zA-Z]+", "", token)
        # remove empty value and single character value
        if new_token != "" and len(new_token) >= 2:
            vowels = len([v for v in new_token if v in "aeiou"])
            if vowels != 0:  # remove line that only contains consonants
                clean_token.append(new_token)

    # Get the list of stop words
    stop_words = stopwords.words("english")
    # add new stopwords to the list
    stop_words.extend(["could", "though", "would", "also", "many", "much"])
    # Remove the stopwords from the list of tokens
    tokens = [x for x in clean_token if x not in stop_words]

    data_tagset = nltk.pos_tag(tokens)
    df_tagset = pd.DataFrame(data_tagset, columns=["Word", "Tag"])

    # Create lemmatizer object
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each word and display the output
    lemmatize_text = []
    for word in tokens:
        output = [
            word,
            lemmatizer.lemmatize(word, pos="n"),
            lemmatizer.lemmatize(word, pos="a"),
            lemmatizer.lemmatize(word, pos="v"),
        ]
        lemmatize_text.append(output)
    # create DataFrame using original words and their lemma words
    df = pd.DataFrame(
        lemmatize_text,
        columns=["Word", "Lemmatized Noun", "Lemmatized Adjective", "Lemmatized Verb"],
    )

    df["Tag"] = df_tagset["Tag"]

    # replace with single character for simplifying
    df = df.replace(["NN", "NNS", "NNP", "NNPS"], "n")
    df = df.replace(["JJ", "JJR", "JJS"], "a")
    df = df.replace(["VBG", "VBP", "VB", "VBD", "VBN", "VBZ"], "v")
    # '''
    # define a function where take the lemmatized word when tagset is noun, and take lemmatized adjectives when tagset is adjective
    # '''
    df_lemmatized = df.copy()
    df_lemmatized["Tempt Lemmatized Word"] = (
        df_lemmatized["Lemmatized Noun"]
        + " | "
        + df_lemmatized["Lemmatized Adjective"]
        + " | "
        + df_lemmatized["Lemmatized Verb"]
    )
    df_lemmatized.head(5)
    lemma_word = df_lemmatized["Tempt Lemmatized Word"]
    tag = df_lemmatized["Tag"]
    i = 0
    new_word = []
    while i < len(tag):
        words = lemma_word[i].split("|")
        if tag[i] == "n":
            word = words[0]
        elif tag[i] == "a":
            word = words[1]
        elif tag[i] == "v":
            word = words[2]
        new_word.append(word)
        i += 1
    df_lemmatized["Lemmatized Word"] = new_word

    # calculate frequency distribution of the tokens
    lemma_word = [str(x).strip() for x in df_lemmatized["Lemmatized Word"]]
    return lemma_word


input_text = "the patient has spasm-like movements. There is concern for subdural hematomas on the CT, so we need one for better visualization"

vals = process_text(input_text)
input_text = " ".join([*set(vals)])

# print(input_text)
