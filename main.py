from unidecode import unidecode
import re
from nltk import *   

def tokenizar(fichero):
    ps=PorterStemmer()
    file = open(fichero,encoding='utf-8')
    stop_words = open("Archivos/spanish.txt",encoding='utf-8')
    stop_words = stop_words.read()
    stop_words = unidecode(stop_words).split("\n")
    
    
    
    lineas = unidecode(file.read())
    sin_Numeros = re.sub("\d+","",lineas)
    
    palabras = re.split('[, .()#&%+:;\n"]',sin_Numeros)
    lista=[]
    for i in palabras:
        i =i.lower()
        if(len(i)>0):
            if(i not in stop_words and i != "[" and i !="]" and i !="[]"):
             lista.append(ps.stem(i,True))
    
    dic_Orden = [Counter(lista).elements()]
    
    
   
tokenizar("Archivos/texto3_Corpus.txt")