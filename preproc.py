import re
import requests
import pandas as pd
from itertools import chain
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary, StopWordRemover


class Preproc(object):
    def __init__(self, data):
        self.data = data
        self.data_normalized = self.normalization()

    def normalization(self):
        hasil_norm = []
        for kalimat in self.data:
            headers = {'x-api-key': 'lSJA4ZycJR14dUQjMsDnHsN3KA1pewNqao91ryjj', 'Content-Type': 'application/json'}
            body = {'text': kalimat}
            r_pos = requests.post('https://api.prosa.ai/v1/normals', headers=headers, json=body)
            kalimat = r_pos.json()['text']
            hasil_norm.append(kalimat)
        # print("Hasil Normalisasi :\n", hasil_norm)
        return hasil_norm
    
    def preprocessing(self):
        #Cleaning
        hasil_cleaning = []

        for kalimat in self.data_normalized:
            kalimat = kalimat.lower()
            kalimat = re.sub(r"http\S+", "", kalimat)
            kalimat = re.sub(r'@\S*\s?|#\S*\s?', ' ', kalimat)
            kalimat = re.sub(r'[0-9]', ' ', kalimat)
            kalimat = re.sub(r'[\!\"\”\$\%\&\'\(\)\*\+\,\-\.\ / \:\;\ < \=\ > \?\[\\\]\ ^ \_\`\{\ | \}\~]', ' ',kalimat)
            hasil_cleaning.append(kalimat)
        # print("Hasil Cleaning :\n", hasil_cleaning)

        #Filtering
        hasil_filtering = []
        read_word_list = pd.read_csv('stopword_list.txt')
        word = list(chain.from_iterable(read_word_list.values))
        factory = StopWordRemoverFactory().get_stop_words()
        data = factory + word
        word_list = ArrayDictionary(data)
        stopword = StopWordRemover(word_list)

        for kalimat in hasil_cleaning:
            kalimat = stopword.remove(stopword.remove(kalimat))
            hasil_filtering.append(kalimat)
        # print("Hasil Filtering :\n",hasil_filtering)

        #Stemming
        hasil_stemming = []
        factory = StemmerFactory()
        stemmer =  factory.create_stemmer()

        for kalimat in hasil_filtering:
            hasil_stemming.append(stemmer.stem(kalimat))
        # print("Hasil Stemming :\n", hasil_stemming)

        #Tokenization
        hasil_token = []
        for kalimat in hasil_stemming:
            # hasil_token.append(word_tokenize(kalimat))
            rx = r"[^()\s]+|[()]"
            hasil_token.append(re.findall(rx, kalimat))
        # print("Hasil Tokenization :\n",hasil_token)

        return hasil_token