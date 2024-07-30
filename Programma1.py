import sys
import nltk

def EstraiStatistiche(frasi):
    testoTokenizzato = []
    testoPOS = []
    nCaratteri = 0
    nTokensPunteggiatura = 0
    
    for frase in frasi:
        tokens = nltk.word_tokenize(frase)
        tokensPOS = nltk.pos_tag(tokens)
        testoTokenizzato += tokens
        testoPOS += tokensPOS
        for tok in tokens: #Scorro tutti i token di ogni frase
            if (tok not in [",",".",":",";","(",")","!","?","'"]): #controllo se quel token è un segno di punteggiatura
                nCaratteri += len(tok) #se non lo è, sommo il numero dei suoi caratteri
            else:
                nTokensPunteggiatura+=1 #altrimenti, conto il numero di token punteggiatura.

    nFrasi = len(frasi)
    nTokens = len(testoTokenizzato)
    nTokensSenzaPunteggiatura = nTokens-nTokensPunteggiatura

    LMediaFrasi = nTokens/nFrasi
    LMediaTokens = nCaratteri/nTokensSenzaPunteggiatura
                
    return testoTokenizzato, testoPOS, nFrasi, nTokens, LMediaFrasi, LMediaTokens

def Hapax1000(tokens): #Calcola il numero di Hapax sui primi 1000 token
    tokens1000 = tokens[0:1000] #Prendo in considerazione solo i primi 1000 token del corpus
    vocabolario1000 = list(set(tokens1000)) #Calcolo il vocabolario dei primi 1000 token (non considero i token ripetuti)
    nHapax = 0
    for tok in vocabolario1000:
        freq = tokens.count(tok)
        if(freq == 1):
            nHapax+=1
            
    return nHapax

def Vocabolario_TTR(tokens): #Stampo il vocabolario e la TTR incrementale (500 tokens alla volta)
    for i in range(0, len(tokens), 500): #La i viene incrementata di 500 ogni ciclo
        tokens500 = tokens[0:i+500]
        vocabolario500 = list(set(tokens500))
        ttr500 = len(vocabolario500)/len(tokens500)

        print("Intervallo 0 -", i+500, "\tVocabolario: ", len(vocabolario500), "\tTTR:", ttr500)

def DistribuzioneSemantica(tokensPOS): #Calcolo la percentuale di parole piene e di parole funzionali
    nParolePiene = 0
    nParoleFunzionali = 0
    for tok in tokensPOS:
        if(tok[1][0] in ["J","N","R","V"] and tok[1] != "RP"): #Verifico la categoria grammaticale controllando il primo carattere di ogni tag POS (non considero RP come un avverbio)
            nParolePiene +=1
        elif(tok[1] in ["DT","IN","CC","PRP","PRP$","WP","WP$"]): #Non posso verificare come sopra per ambiguità nelle prime lettere dei tag che mi interessano
            nParoleFunzionali += 1
            
    #Calcolo le percentuali
    percParolePiene = nParolePiene/len(tokensPOS)*100
    percParoleFunzionali = nParoleFunzionali/len(tokensPOS)*100
    return percParolePiene, percParoleFunzionali
        
def main(file1, file2):
    inputFile1 = open(file1, mode="r", encoding="utf-8")
    inputFile2 = open(file2, mode="r", encoding="utf-8")
    testo1 = inputFile1.read()
    testo2 = inputFile2.read()

    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    frasiTesto1 = sent_tokenizer.tokenize(testo1)
    frasiTesto2 = sent_tokenizer.tokenize(testo2)

    testoTokenizzato1, testoPOS1, nFrasiTesto1, nTokenTesto1, LMediaFrasiTesto1, LMediaTokensTesto1 = EstraiStatistiche(frasiTesto1)
    testoTokenizzato2, testoPOS2, nFrasiTesto2, nTokenTesto2, LMediaFrasiTesto2, LMediaTokensTesto2 = EstraiStatistiche(frasiTesto2)

    nHapax1000Testo1 = Hapax1000(testoTokenizzato1)
    nHapax1000Testo2 = Hapax1000(testoTokenizzato2)

    percParolePiene1, percParoleFunzionali1 = DistribuzioneSemantica(testoPOS1)
    percParolePiene2, percParoleFunzionali2 = DistribuzioneSemantica(testoPOS2)

    #Stampo i dati raccolti
    
    print("Il corpus", file1, "contiene:", nFrasiTesto1, "frasi e ", nTokenTesto1, "token.")
    print("La lunghezza media delle frasi è di", LMediaFrasiTesto1, "token per frase; la lunghezza media dei token è:", LMediaTokensTesto1, "caratteri per token.")

    print("\nIl corpus", file2, "contiene:", nFrasiTesto2, "frasi e ", nTokenTesto2, "token.")
    print("La lunghezza media delle frasi è di", LMediaFrasiTesto2, "token per frase; la lunghezza media dei token è:", LMediaTokensTesto2, "caratteri per token.")

    print("\nNumero di Hapax nei primi 1000 token del corpus", file1,":", nHapax1000Testo1)
    print("Numero di Hapax nei primi 1000 token del corpus", file2, ":", nHapax1000Testo2)
    
    #La stampa avviene dentro la funzione
    print("\nVocabolario e TTR incrementale del corpus", file1)
    Vocabolario_TTR(testoTokenizzato1) 
    print("\nVocabolario e TTR incrementale del corpus", file2)
    Vocabolario_TTR(testoTokenizzato2)

    print("\nIl corpus", file1, "è costituito da:", percParolePiene1, "% parole piene e", percParoleFunzionali1, "% di parole funzionali")
    print("\nIl corpus", file2, "è costituito da:", percParolePiene2, "% parole piene e", percParoleFunzionali2, "% di parole funzionali")

main(sys.argv[1], sys.argv[2])
