from unidecode import unidecode
import re
import nltk
from nltk import *
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy


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


def representar(vectores, similares):
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


def embeddings(corpus, vectores):
    corpus_embeddings = []
    for c in corpus:
        # Obtener el embedding de cada palabra en la frase y promediarlos
        embeddings = [vectores[word] for word in c if word in vectores]
        if embeddings:
            sentence_embedding = numpy.mean(embeddings, axis=0)
            corpus_embeddings.append(sentence_embedding)
    return corpus_embeddings


if __name__ == "__main__":
    # Corpus
    corpus1 = tokenizar("archivos/texto1_corpus.txt")
    corpus2 = tokenizar("archivos/texto2_corpus.txt")
    corpus3 = tokenizar("archivos/texto3_corpus.txt")
    corpus4 = tokenizar("archivos/texto4_corpus.txt")

    corpus_total = corpus1 + corpus2 + corpus3 + corpus4

    # Entrenar el modelo Word2Vec
    model1 = Word2Vec(sentences=corpus1, vector_size=100, window=5, min_count=1, workers=4)
    model2 = Word2Vec(sentences=corpus2, vector_size=100, window=5, min_count=1, workers=4)
    model3 = Word2Vec(sentences=corpus3, vector_size=100, window=5, min_count=1, workers=4)
    model4 = Word2Vec(sentences=corpus4, vector_size=100, window=5, min_count=1, workers=4)

    # Obtener vectores de palabras
    word_vectors1 = model1.wv
    word_vectors2 = model2.wv
    word_vectors3 = model3.wv
    word_vectors4 = model4.wv

    word_vectors_total = Word2VecKeyedVectors(vector_size=100)
    word_vectors_total.add_vectors(word_vectors1.index_to_key, word_vectors1.vectors)
    word_vectors_total.add_vectors(word_vectors2.index_to_key, word_vectors2.vectors)
    word_vectors_total.add_vectors(word_vectors3.index_to_key, word_vectors3.vectors)
    word_vectors_total.add_vectors(word_vectors4.index_to_key, word_vectors4.vectors)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    tematicas = ['matematicas'] * len(corpus1) + ['salud'] * len(corpus2) + ['lengua'] * len(corpus3) + ['filosofia'] * len(corpus4)
    total_embeddings = embeddings(corpus_total, word_vectors1)
    X_train, X_test, y_train, y_test = train_test_split(total_embeddings, tematicas, test_size=0.6, random_state=42)

    # Entrenar un modelo de clasificación
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el rendimiento del modelo
    print(classification_report(y_test, y_pred))

