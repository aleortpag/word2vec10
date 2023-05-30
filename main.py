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
   lista_final = []
   palabras = re.split('[, .()#&%+:;\n"]',sin_Numeros)
   lista=[]
   for i in palabras:
      i =i.lower()
      if(len(i)>0):
         if(i not in stop_words and i != "[" and i !="]" and i !="[]"):
            lista.append(ps.stem(i,True))
   return lista

def contar_palabras(lista):
   frecuencia = {}
   for palabra in lista:
      frecuencia[palabra] = frecuencia.get(palabra, 0)+1
   lista_tuplas = [(palabra, contador) for palabra, contador in frecuencia.items()]
   return sorted(lista_tuplas,reverse=True,key=lambda x: x[1])

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
if __name__ == '__main__':
    lista_palabras = tokenizar("archivos/texto1_Corpus.txt")
    lista_tuplas = contar_palabras(lista_palabras)
    lista_tuplas = eliminar_menores_de(lista_tuplas, 10)
    palabras = elimina_pairs(lista_tuplas)
    print(lista_tuplas)
    print(palabras)
######
