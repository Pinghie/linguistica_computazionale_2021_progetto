import sys
import nltk
from math import log2 #Importo il logaritmo in base due per il calcolo della LMI

def ordina(dizionario): #Ordina il dizionario in ordine decrescente per valore
    return sorted(dizionario.items(), key=lambda x: x[1], reverse = True)

def AnnotazioneLinguistica(frasi):
    testoTokenizzato = []
    testoPOS = []
    listaNomiPersona = [] #Questa lista conterrà tutti i nomi propri di persona del corpus
    
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        analisiNE = nltk.ne_chunk(tokensPOS)
        
        for nodo in analisiNE:
            NE=''
            if hasattr(nodo, "label"):
               if nodo.label() == "PERSON":
                   for partNE in nodo.leaves():
                       NE = NE + " " + partNE[0]
                   listaNomiPersona.append(NE) #Aggiungo alla lista tutti i nomi propri di persona che incontro
        
        testoTokenizzato += tokens
        testoPOS += tokensPOS
        
    return testoTokenizzato, testoPOS, listaNomiPersona       

def frequenzePOS(testoPOS):
    listaPOS = []
    aggettiviPOS = []
    avverbiPOS = []
    
    for coppiaPOS in testoPOS:
        listaPOS.append(coppiaPOS[1]) #Creo una lista con tutte le POS
        if(coppiaPOS[1][0] == "J"):
            aggettiviPOS.append(coppiaPOS) #Creo una lista con tutti gli aggettivi
        elif(coppiaPOS[1][0] == "R" and coppiaPOS[1]!="RP"):
            avverbiPOS.append(coppiaPOS) #Creo una lista con tutti gli avverbi

    #Calcolo e stampo le 10 POS più frequenti
    distPOS = nltk.FreqDist(listaPOS)
    distPOS10_ordinata = distPOS.most_common(10)
    print("\nLe dieci Part of Speech più frequenti")
    for pos in distPOS10_ordinata:
        print(pos[0], "\t\tFrequenza:", pos[1])

    print()

    #Calcolo e stampo i 10 bigrammi POS più frequenti
    bigrammiPOS = nltk.bigrams(listaPOS)
    distBigrammiPOS = nltk.FreqDist(bigrammiPOS)
    distBigrammiPOS10_ordinata = distBigrammiPOS.most_common(10)
    print("I dieci bigrammi Part of Speech più frequenti")
    for pos in distBigrammiPOS10_ordinata:
        print(pos[0][0], "-", pos[0][1], "\t\tFrequenza:", pos[1])

    print()

    #Calcolo e stampo i 10 trigrammi POS più frequenti
    trigrammiPOS = nltk.trigrams(listaPOS)
    distTrigrammiPOS = nltk.FreqDist(trigrammiPOS)
    distTrigrammiPOS10_ordinata = distTrigrammiPOS.most_common(10)
    print("I dieci trigrammi Part of Speech più frequenti")
    for pos in distTrigrammiPOS10_ordinata:
        print(pos[0][0], "-", pos[0][1], "-", pos[0][2], "\t\tFrequenza:", pos[1])

    print()

    #Calcolo e stampo i 20 aggettivi più frequenti
    distAggettiviPOS = nltk.FreqDist(aggettiviPOS)
    distAggettiviPOS20_ordinata = distAggettiviPOS.most_common(20)
    print("I venti aggettivi più frequenti")
    for pos in distAggettiviPOS20_ordinata:
        print(pos[0][0], "\t\tFrequenza:", pos[1])

    print()

    #Calcolo e stampo i 20 avverbi più frequenti
    distAvverbiPOS = nltk.FreqDist(avverbiPOS)
    distAvverbiPOS20_ordinata = distAvverbiPOS.most_common(20)
    print("I venti avverbi più frequenti")
    for pos in distAvverbiPOS20_ordinata:
        print(pos[0][0], "\t\tFrequenza:", pos[1])

def BigrAggSost(testoPOS, testoTokenizzato):
    bigrammiPOS = list(nltk.bigrams(testoPOS))
    bigrammiAggSost = [] #Questa lista conterrà tutti i bigrammi Agg Sost
    
    for bigramma in bigrammiPOS:
        if(bigramma[0][1] in ["JJ", "JJR", "JJS"] and bigramma[1][1] in ["NN", "NNS", "NNP", "NNPS"] and testoTokenizzato.count(bigramma[0][0])>3 and testoTokenizzato.count(bigramma[1][0])>3):
            bigrammiAggSost.append(bigramma)

    #Calcolo la distribuzione di frequenza dei bigrammi Agg Sost
    distAggSost = nltk.FreqDist(bigrammiAggSost)
    distAggSost_ordinata = distAggSost.most_common(20) #Calcolo i 20 bigrammi con frequenza massima (nella lista che contiene solo i bigrammi Agg Sost)
    
    uniqBigrammiAggSost = list(set(bigrammiAggSost)) #Questa lista contiene i bigrammi agg-sost POS unici
        
    #Calcolo la probabilità condizionata di ogni bigramma Agg Sost
    dizBigrammiProbCond={} #Questo dizionario conterrà tutti i bigrammi POS con relativa Prob Cond
    for bigramma in uniqBigrammiAggSost:
        freqBigramma = bigrammiPOS.count(bigramma)
        token1 = bigramma[0][0]
        freqToken1 = testoTokenizzato.count(token1) #Frequenza del primo elemento del bigramma
        dizBigrammiProbCond[bigramma] = freqBigramma/freqToken1
    listaBigrammiPerProbCond = ordina(dizBigrammiProbCond) #Ordino il dizionario in ordine decrescente per valore (in questo caso, per probabilità condizionata)
        
    #Calcolo la local mutual information di ogni bigramma Agg Sost
    dizBigrammiLMI={} #Questo dizionario conterrà tutti i bigrammi POS con relativa LMI
    for bigramma in uniqBigrammiAggSost:
        freqBigramma = bigrammiPOS.count(bigramma)
        N = len(testoTokenizzato) #Lunghezza del Corpus
        token1 = bigramma[0][0]
        freqToken1 = testoTokenizzato.count(token1) #Frequenza del primo elemento del bigramma
        token2 = bigramma[1][0]
        freqToken2 = testoTokenizzato.count(token2) #Frequenza del secondo elemento del bigramma
        dizBigrammiLMI[bigramma] = freqBigramma * log2((freqBigramma*N)/(freqToken1*freqToken2))
    listaBigrammiPerLMI = ordina(dizBigrammiLMI) #Ordino il dizionario in ordine decrescente per valore (in questo caso, per LMI)

    #Stampo i risultati
    print("\nI venti bigrammi composti da aggettivo e sostantivo con frequenza massima:")
    for bigramma in distAggSost_ordinata:
        print(bigramma[0][0][0], "-", bigramma[0][1][0], "\t\tFrequenza:", bigramma[1])

    print("\nI venti bigrammi composti da aggettivo e sostantivo con probabilità condizionata massima:")
    for bigramma in listaBigrammiPerProbCond[0:20]:
        print(bigramma[0][0][0], "-", bigramma[0][1][0], "\t\tProbabilità condizionata:", bigramma[1])

    print("\nI venti bigrammi composti da aggettivo e sostantivo con Local Mutual Information massima:")
    for bigramma in listaBigrammiPerLMI[0:20]:
        print(bigramma[0][0][0], "-", bigramma[0][1][0], "\t\tLMI:", bigramma[1])

def mediaDistFreqFrase(frasi, testo):
    maxMedia = 0
    minMedia = 0 #La media non può mai essere zero: uso questo valore per assegnare il valore medio minimo alla prima frase accettabile del corpus (vedi riga A)
    probMax = 0
    bigrammiTesto = list(nltk.bigrams(testo))
    trigrammiTesto = list(nltk.trigrams(testo))

    for frase in frasi:
        sommaFreq = 0 #Questa variabile conterrà la somma delle frequenze dei token della frase
        flag = True #Mi serve per verificare che ogni token della frase abbia una frequenza all'interno del corpus >2
        tokens = nltk.word_tokenize(frase)
        l = len(tokens) #Calcolo il numero di token nella frase

        if (l>=6 and l<25):
            
            for token in tokens:
                if(testo.count(token)>=2):
                    sommaFreq += testo.count(token) #Se il token di una frase ha frequenza maggiore di 2, sommo la sua frequenza
                else:
                    flag = False
                    break
            if(flag == False):
                continue #Al primo token che trovo con frequenza minore di 2, smetto di controllare le altre parole della frase, e passo alla frase successiva.

            #Se arrivo a questo punto del codice, sono sicuro che la frase non ha token con frequenza <2
            mediaDistFreq = sommaFreq/l
            if(maxMedia < mediaDistFreq):
                maxMedia = mediaDistFreq
                fraseMaxMedia = frase
            if(minMedia > mediaDistFreq or minMedia == 0): #(A)
                minMedia = mediaDistFreq
                fraseMinMedia = frase
            
            #Calcolo la probabilità di ogni frase applicando un modello markoviano di ordine II
            trigrammiFrase = list(nltk.trigrams(tokens))
            primoToken = trigrammiFrase[0][0]
            prob = testo.count(primoToken)/len(testo) #Calcolo il primo fattore della moltiplicazione per calcolare la probabilità
            primoBigramma = list(nltk.bigrams(trigrammiFrase[0]))[0] #Ricavo il primo bigramma, estaendo il primo dei bigrammi del trigramma
            prob *= bigrammiTesto.count(primoBigramma)/testo.count(primoToken) #Calcolo il secondo fattore della moltiplicazione per calcolare la probabilità e lo moltiplico al precedente
            for trigramma in trigrammiFrase:
                freqTrigramma = trigrammiTesto.count(trigramma)
                bigrammaInterno = list(nltk.bigrams(trigramma))[0]
                freqBigramma = bigrammiTesto.count(bigrammaInterno)

                prob *= freqTrigramma/freqBigramma
            
            if(probMax < prob):
                probMax = prob
                fraseProbMax = frase

    #Stampo i risultati ottenuti                            
    print("La frase con media della distribuzione di frequenza dei token più alta è:", fraseMaxMedia, "con distribuzione media di", maxMedia)
    print("La frase con media della distribuzione di frequenza dei token più bassa è:", fraseMinMedia, "con distribuzione media di", minMedia)
    print("La frase con probabilità più alta secondo il modello markoviano di II ordine è:", fraseProbMax, "con probabilità ", probMax)


def nomiPropriPiuFrequenti(listaNomiPersona):
    distNomiPropri = nltk.FreqDist(listaNomiPersona)
    top15NomiPropri = distNomiPropri.most_common(15)
    for elem in top15NomiPropri:
        print(elem[0], "\tFrequenza:", elem[1]) 
    
def main(file1, file2):
    inputFile1 = open(file1, mode="r", encoding="utf-8")
    inputFile2 = open(file2, mode="r", encoding="utf-8")
    testo1 = inputFile1.read()
    testo2 = inputFile2.read()

    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    frasiTesto1 = sent_tokenizer.tokenize(testo1)
    frasiTesto2 = sent_tokenizer.tokenize(testo2)
    
    testoTokenizzato1, testoPOS1, listaNomiPersona1 = AnnotazioneLinguistica(frasiTesto1)
    testoTokenizzato2, testoPOS2, listaNomiPersona2 = AnnotazioneLinguistica(frasiTesto2)
    
    print("Distribuzioni POS del corpus", file1)
    frequenzePOS(testoPOS1)
    print("\n\nDistribuzioni POS del corpus", file2)
    frequenzePOS(testoPOS2)
    
    print("\n\nAnalisi dei 20 bigrammi Aggettivo-Sostantivo del corpus", file1)
    BigrAggSost(testoPOS1, testoTokenizzato1)
    print("\n\nAnalisi dei 20 bigrammi Aggettivo-Sostantivo del corpus", file2)
    BigrAggSost(testoPOS2, testoTokenizzato2)
  
    print("\n\nAnalisi delle frasi del corpus", file1)
    mediaDistFreqFrase(frasiTesto1, testoTokenizzato1)
    print("\n\nAnalisi delle frasi del corpus", file2)
    mediaDistFreqFrase(frasiTesto2, testoTokenizzato2)
    
    print("\n\nI nomi propri di persona più frequenti del corpus", file1)
    nomiPropriPiuFrequenti(listaNomiPersona1)
    print("\n\nI nomi propri di persona più frequenti del corpus", file2)
    nomiPropriPiuFrequenti(listaNomiPersona2)
    

main(sys.argv[1], sys.argv[2])
