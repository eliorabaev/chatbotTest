import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import load_model
import tkinter as tk
from tkinter import ttk

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send():
    message = entry_field.get()
    ints = predict(message)
    res = response(ints, intents)
    chat.insert(tk.END, "You: " + message + '\n\n')
    chat.insert(tk.END, "Bot: " + res + '\n\n')
    entry_field.delete(0, tk.END)

root = tk.Tk()
root.title("Chatbot")

frame = tk.Frame(root)
scrollbar = tk.Scrollbar(frame)
chat = tk.Listbox(frame, width=150, height=20, yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat.pack(side=tk.LEFT, fill=tk.BOTH, pady=10)
frame.pack()

entry_field = ttk.Entry(root, width=60)
entry_field.pack(fill=tk.X, padx=10)

send_button = ttk.Button(root, text="Send", command=send)
send_button.pack(ipadx=20)

root.mainloop()
