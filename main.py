from unidecode import unidecode
import re
import nltk
from nltk import *
from gensim.models import Word2Vec


def tokenizar(fichero):
    ps = PorterStemmer()
    file = open(fichero, encoding='utf-8')
    stop_words = open("Archivos/spanish.txt", encoding='utf-8')
    stop_words = stop_words.read()
    stop_words = unidecode(stop_words).split("\n")

    lineas = unidecode(file.read())
    sin_Numeros = re.sub("\d+", "", lineas)

    palabras = re.split('[, .()#&%+:;\n"]', sin_Numeros)
    lista = []
    for i in palabras:
        i = i.lower()
        if (len(i) > 0):
            if (i not in stop_words and i != "[" and i != "]" and i != "[]"):
                lista.append(ps.stem(i, True))
    return lista


def contar_palabras(lista):
    frecuencia = {}
    for palabra in lista:
        frecuencia[palabra] = frecuencia.get(palabra, 0)+1
    lista_tuplas = [(palabra, contador)
                    for palabra, contador in frecuencia.items()]
    return sorted(lista_tuplas, reverse=True, key=lambda x: x[1])


def eliminar_menores_de(lista, umbral):
    res = []
    for i in lista:
        if i[1] > umbral:
            res.append(i)
    return res


def elimina_pairs(lista_tuplas):
    res = []
    for i in lista_tuplas:
        res.append(i[0])
    return res


##### main #####

if __name__ == "__main__":

    with open("archivos/texto1_corpus.txt", "r", encoding="utf-8") as file:
        texto = file.read()

    # Tokenizar el texto en frases
    frases = re.sub("\d","",unidecode(texto))
    frases = re.sub("\.","",frases)
    frases = nltk.sent_tokenize(frases)


    # Tokenizar cada frase en palabras
    corpus = [word_tokenize(frase) for frase in frases]

    stop_words = open("Archivos/spanish.txt", encoding='utf-8')
    stop_words = stop_words.read()
    stop_words = unidecode(stop_words).split("\n")

    lista = ["!", "@", "#", "$", "%", "&", "*", "(", ")", "-", "", "+", "=", "{", "}", "[", "]", "|", "'\'", ";", ":", ",", ".", "<", ">", "/",
             "?", "¡", "¿", "¨", "´", "¨", "~", "¬", "`", "^", "¦", "ª", "º", "«", "»", "©", "®", "™", "§", "±", "µ", "÷", "×", "``", "\d"]

    res = []

    for i in corpus:
        res1 = []
        for j in i:
            if j.lower() not in stop_words and j not in lista and len(j) > 1:
                res1.append(j.lower())
        res.append(res1)

    # Entrenar el modelo Word2Vec
    model = Word2Vec(sentences=res, vector_size=100,
                     window=5, min_count=1, workers=4)

    # Obtener vectores de palabras
    word_vectors = model.wv

    # Realizar operaciones con los vectores
    similar_words = word_vectors.similar_by_vector(
        word_vectors.vectors.mean(axis=0), topn=15)

    # similarity = word_vectors.similarity("cero", "de")
    print(similar_words)
######
