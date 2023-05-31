from unidecode import unidecode
import re
import nltk
from nltk import *
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
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
    return res"""

LISTA_NO_ALFANUMERICOS = ["!", "@", "#", "$", "%", "&", "*", "(", ")", "-", "", "+", "=", "{", "}", "[", "]", "|", "'\'", ";", ":", ",", ".", "<", ">", "/",
                          "?", "¡", "¿", "¨", "´", "¨", "~", "¬", "`", "^", "¦", "ª", "º", "«", "»", "©", "®", "™", "§", "±", "µ", "÷", "×", "``", "\d"]


def tokenizar(archivo):
    with open(archivo, "r", encoding="utf-8") as file:
        texto = file.read()

    # Tokenizar el texto en frases
    frases = re.sub("\d", "", unidecode(texto))
    frases = re.sub("\.", "", frases)
    frases = nltk.sent_tokenize(frases)

    # Tokenizar cada frase en palabras
    corpus = [word_tokenize(frase) for frase in frases]

    stop_words = open("Archivos/spanish.txt", encoding='utf-8')
    stop_words = stop_words.read()
    stop_words = unidecode(stop_words).split("\n")

    res = []

    for i in corpus:
        res1 = []
        for j in i:
            if j.lower() not in stop_words and j not in LISTA_NO_ALFANUMERICOS and len(j) > 1:
                res1.append(j.lower())
        res.append(res1)
    return res


def representa(vectores, similares):
    # Reducción de dimensionalidad a 2 dimensiones
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectores.vectors)

    # Seleccionar palabras de interés
    words_of_interest = [word for word, _ in similares]
    vectors_of_interest = [vectores[word] for word in words_of_interest]

    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)

    # Anotar las palabras de interés
    for i, word in enumerate(words_of_interest):
        plt.annotate(
            word, xy=(vectors_of_interest[i][0], vectors_of_interest[i][1]), fontsize=12)

    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.title('Visualización de Vectores de Palabras')
    plt.show()


if __name__ == "__main__":
    corpus = tokenizar("archivos/texto1_corpus.txt")

    # Entrenar el modelo Word2Vec
    model = Word2Vec(sentences=corpus, vector_size=100,
                     window=5, min_count=1, workers=4)

    # Obtener vectores de palabras
    word_vectors = model.wv

    # Realizar operaciones con los vectores
    similar_words = word_vectors.similar_by_vector(
        word_vectors.vectors.mean(axis=0), topn=15)
    
    representa(word_vectors, similar_words)

    # similarity = word_vectors.similarity("cero", "de")
    print(similar_words)
