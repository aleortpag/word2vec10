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

LISTA_NO_ALFANUMERICOS = ["!", "@", "#", "$", "%", "&", "*", "(", ")", "-", "", "+", "=", "{", "}", "[", "]", "|", "'", "\\", ";", ":", ",", ".", "<", ">", "/", "?", "¡", "¿", "¨", "´", "¨", "~", "¬", "`", "^", "¦", "ª", "º", "«", "»", "©", "®", "™", "§", "±", "µ", "÷", "×", "``", "\\d", "-"]

def tokenizar(archivo):
    with open(archivo, "r", encoding="utf-8") as file:
        texto = file.read()

    # Tokenizar el texto en frases
    frases = re.sub("\\d", "", unidecode(texto))
    frases = re.sub("\\.", "", frases)
    frases = nltk.sent_tokenize(frases)

    # Tokenizar cada frase en palabras
    corpus = [word_tokenize(frase) for frase in frases]

    stop_words = open("stop_words.txt", encoding='utf-8')
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

def add_text_to_corpus(corpus,tematica,tematicas):
    new_corpus = tokenizar("corpus/"+tematica+".txt")
    corpus += new_corpus
    tematicas.append(tematica)
    return corpus, tematicas


def train_word2vec(corpus):
    model = Word2Vec(
        corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
    return model

def main():
    # Carpeta donde se encuentran los archivos de texto
    folder_path = "corpus"

    # Corpus inicial y temáticas
    corpus = []
    tematicas = []

    # Añadir textos al corpus
    for filename in os.listdir(folder_path):
      if filename.endswith(".txt"):
        tematica = filename.split(".txt")[0]
        corpus, tematicas = add_text_to_corpus(corpus, tematica, tematicas)


    # Entrenar el modelo Word2Vec
    model = train_word2vec(corpus)

    # Visualizar palabras similares
    word = "lengua"
    similares = model.wv.most_similar(word)
    representar(model.wv, similares)

    # Obtener representaciones de embeddings
    test_file = "test/test3.txt"
    test_corpus = tokenizar(test_file)
    test_embeddings = embeddings(test_corpus, model.wv)
    total_embeddings = embeddings(corpus, model.wv)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train = total_embeddings
    y_train = tematicas

    # Entrenar un modelo de clasificación
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    resultado_clasificacion = model.predict(test_embeddings)
    print("El resultado de la clasificación fue: " + resultado_clasificacion[0])

    # Evaluar el rendimiento del modelo

if __name__ == "__main__":
    main()
