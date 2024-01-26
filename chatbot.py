import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def get_question(intents_list, intents_json):
    question = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == question:
            ask = random.choice(i['questions'])
            break
    return ask


while True:
    print("\nHey I am kyra, your personal Medbot to help with your health issues.")
    print("what seems to be your symptoms? ")
    message = input("")
    if message == "Bye" or message == "bye":
        print("Have a great day")
        break
    ints = predict_class(message)
    diagnose = get_question(ints, intents)
    print(diagnose)
    ans = input("Enter y(for yes) or n(for no): ")

    if ans == "n" or ans == "N":
        ans2 = "n"
        while ans2 != "y" or ans2 != "Y":
            print("\ni`m sorry could you tell me your symptoms more accurately?")
            message2 = input("")
            ints2 = predict_class(message2)
            diagnose2 = get_question(ints2, intents)
            res2 = get_response(ints2, intents)
            print(diagnose2)
            ans2 = input("Enter y(for yes) or n(for no): ")
            if ans2 == "y" or ans2 == "Y":
                print(res2)
                print('I cannot end our session until you tell me "Bye"')
                break

    if ans == "y" or ans =="Y":
        res = get_response(ints, intents)
        print(res)
        print('I cannot end our session until you tell me "Bye"')
