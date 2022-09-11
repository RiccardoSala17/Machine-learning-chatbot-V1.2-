# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:01:45 2022

@author: Riccardo Sala
"""

'''
Language of chatbot: ENG  Comments: ITA/ENG

ENG: Simple ML/NLP chatbot decoding/encoding technology based.
ITA: Semplice chatbot ML/NLP basato su tecnologia decoding/encoding.

ENG: Dataset source: Cornell movie dialog corpus               Link: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
ITA: Dataset di riferimento: Cornell movie dialog corpus
'''

#PHASE 1: text pre-processing

'''
ENG: Import Regex module for text matching
ITA: Importiamo il modulo Regex per trovare corrispondenze nel testo
'''

import re

'''
ENG: Open dataset files, we'll ignore all opening errors and split text by lines
ITA: Apriamo i file del dataset, ignoreremo gli errori all'apertura e splittiamo testo riga per riga
'''

lines = open(r'data\movie_lines.txt', encoding='utf-8',
             errors='ignore').read().split('\n')

convers = open(r'data\movie_conversations.txt', encoding='utf-8',
             errors='ignore').read().split('\n')

'''
ENG: Clean the dataset from non-text carachters, we need to create a LIST for conversations and a DICTIONARY
for storing single lines of dialogue
ITA: Puliamo i dataset dai caratteri non testuali, creaiamo una LISTA per conversazioni e un DIZIONARIO per
raggruppare le singole linee di dialogo
'''

exchn = []
for conver in convers:
    exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",","").split())

diag = {}
for line in lines:
    diag[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

'''
ENG: Create lists of question and answer from 'lines' dataset
ITA: Creaiamo liste di domande e risposte dal dataset 'lines'
'''

questions = []
answers = []

for conver in exchn:
    for i in range(len(conver) - 1):        #'-1' to make sure to not have unpaired lines
                                            #'-1' per assicurarci di non avere linee spaiate        
        questions.append(diag[conver[i]])
        answers.append(diag[conver[i+1]])

'''
ENG: Delete the variables we would use no more to reduce memory usage (this is crucial when run on Notebooks)
We will do it several times
ITA: Cancelliamo le variabili che non useremo più per risparmiare memoria (indispensabile per farlo girare su Notebook)
Sarà fatto più volte
'''

del(lines, line, convers, conver, diag, exchn, i)

'''
ENG: Need to filter once more our lines. For training purpose is no good having no limit in lines lenght,
because long paragraphhs and complex concepts may overfit the model. Set max lenght at 13 words
ITA: Bisogna filtra ancora le nostre frasi. Non fa bene al nostro training non avere un limite alla lunghezza
delle frasi, poichè lunghi paragrafi e concetti complessi potrebbero causare overfitting
Fissiamo quindi la lunghezza massima a 13 parole
'''

sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])

'''
ENG: Function 'clean_text' replaces every contracted form in English in order to get only extended forms,
transforms capital letters to lowercase, delete punctuation marks
ITA: La funzione 'clean_text' sostituisce le forme contratte della lingua inglese, in modo da avere solo forme estese,
trasforma le lettere maiuscole in minuscole, cancella la punteggiatura
'''

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


clean_ques = []
clean_ans = []

for line in sorted_ques:
    clean_ques.append(clean_text(line))
        
for line in sorted_ans:
    clean_ans.append(clean_text(line))



# Delete variables
del(answers, questions, line)

'''
ENG: Define max lenght of answers for best fitting
ITA: Definiamo una lunghezza massima anche alle risposte per un miglior risultato
'''

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])


# Delete variables
del(sorted_ans, sorted_ques)

'''
ENG: Divide the dataset to have less data to process (THIS PARAMETER MAY BE CHANGED)
ITA: Tagliamo il dataset per avere meno dati da processare (QUESTO PARAMENTRO PUò ESSERE MODIFICATO)
'''

clean_ans=clean_ans[:20000]
clean_ques=clean_ques[:20000]

'''
ENG: Count word occurrencies in order to create a vocabulary with the most commons
ITA: Contiamo le ripetizioni delle parole, in modo da creare un vocabolario con le più comuni
'''

word2count = {}

for line in clean_ques:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Delete variables
del(word, line)

'''
ENG: Remove the less frequent occurrencies and create the vocabulary, thresh = max occurrencies (THIS PARAMETER MAY BE CHANGED)
ITA: Togliamo le parole meno frequenti e creiamo il vocabolario, thresh = max ripetizioni  (QUESTO PARAMENTRO PUò ESSERE MODIFICATO)
'''

thresh = 5

vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
        
# Delete variables
del(word2count, word, count, thresh)       
del(word_num)        

'''
ENG: Define tokens for encoding/decoding purpose.              SOS = start of string, EOS = end of string, OUT = out of vocabulary, PAD = padded words
ITA: Definiamo i tokens per il meccanismo encoding/decoding
'''

for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

'''
ENG: Append tokens to vocabulary
ITA: Aggiungiamo i tokens al vocabolario
'''

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

'''    
ENG: Give values to unnecessary words like people names, chatbot receives a 0 input value
ITA: Diamo un valore a parole non necessarie come i nomi propri, il chatbot riceverà un input di valore 0   
'''

vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

# Delete variables
del(token, tokens) 
del(x)

'''
ENG: Create inverse vocabulary, reversing key 'w' and value 'v' for decoding purpose
ITA: Creiamo un vocabolario inverso, invertendo key 'w' e valori 'v'
'''

inv_vocab = {w:v for v, w in vocab.items()}



# Delete variables
del(i)

'''
ENG: Create dec/enc inputs, we create a list of word in our vocabulary, if not in vocab we assign key <OUT>
ITA: Creiamo i dec/enc input, creiamo una lista delle parole presenti nel vocabolario, se non presente assegnamo la key <OUT>
'''

encoder_inp = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)

# Delete variables
del(clean_ans, clean_ques, line, lst, word)

'''
ENG: Trasform the list in arrays, we give max lenght(13) and 'post' to padding and truncating to mantain the sequence value
ITA: trasformiamo la lista in array, diamo lunghezza 13 e 'post' to padding and truncating per far mantenere il valore della sequenza
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

'''
ENG: Create output decoding list, taking the array made by decoder_inp without tokens, then pad_sequence output
ITA: Creiamo la lista degli output per il deconding, prendiamo l'array decoder_inp escludendo i tokens, poi applichiamo pad_sequence all'output
'''
    
decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')

# Delete variables
del(i)

'''
ENG: Convert 2d array to 3d matrix for further LSTM use
ITA: convertiamo l'array 2d in una matrice 3d per il futuro uso con LSTM
'''
    
from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, len(vocab))


print("finished prepr")

#END PHASE 1


#PHASE 2: creating model


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

'''
ENG: Instantiate Keras tensors with max lenght as shape
ITA: instanziamo i tensori con la lunghezza massima come shape 
'''
    
enc_inp = Input(shape=(13, ))
dec_inp = Input(shape=(13, ))

'''
ENG: Define embedding layer, reduce output dimension (THIS PARAMETER MAY BE CHANGED)
ITA: Definiemo l'embedding layer, riducendo la la grandezza dell'output (QUESTO PARAMENTRO PUò ESSERE MODIFICATO)
'''
    
VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE+1, output_dim=50, 
                  input_length=13,
                  trainable=True                  
                  )

'''
ENG: Connect enc/dec placeholders to embedding layer (set lstm cell number to 400, THIS PARAMETER MAY BE CHANGED)
set True to paramenters to better monitoring encoding states. Variables 'h' and 'c' store state values
ITA: Connettiamo gli elementi enc/dec all'embedding layer (settiamo il numero di celle LSTM a 400 QUESTO PARAMENTRO PUò ESSERE MODIFICATO)
settiamo True ai parametri per monitorare meglio gli states. Le variabili 'h' e 'c' immagazzinano i valori degli states
'''
    
enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

'''
ENG: we don't need decoding state values, we mantain encoding ones
ITA: non abbiamo bisogno dei valori states nel decoding, quindi manteniamo i valori degli states encoding
'''

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

'''
ENG: Define dense layer and get the final decoder output
ITA: Definiamo il dense layer e  il decoder output finale
'''
    
dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(dec_op)

'''
ENG: Wrap up our final inputs/output in a model using proper Keras functions
ITA: Riuniamo gli input/output finali in un modello con la rispettiva funzione di Keras
'''
    
model = Model([enc_inp, dec_inp], dense_op)

'''
ENG: Compile model with standard parameters and do the fit. Define epochs  (THIS PARAMETER MAY BE CHANGED)
ITA: Compiliamo il modello con parametri standard e diamo il fit. Definiamo il numero delle epoche (QUESTO PARAMENTRO PUò ESSERE MODIFICATO)
'''
    
model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

model.fit([encoder_inp, decoder_inp],decoder_final_output,epochs=150)

print("finish encoder model")

'''
ENG: Create model setup for text predictions
ITA: Settiamo il modello per la previsione del testo
'''
    
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np

enc_model = Model([enc_inp], enc_states)

'''
ENG: Instantiate decoder states inputs using LSTM cells
ITA: istanziamo gli input dei decoder states usanno celle LSTM
'''
    
decoder_state_input_h = Input(shape=(400,))
decoder_state_input_c = Input(shape=(400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

'''
ENG: Define inputs and outputs to use in decoder model (list plus states)
ITA: Definiamo inputs e outputs da usare nel modello di decoding (liste di input/output più gli states)
'''

decoder_outputs, state_h, state_c = dec_lstm(dec_embed , 
                                    initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]


dec_model = Model([dec_inp]+ decoder_states_inputs,
                                      [decoder_outputs]+ decoder_states)

print("finish decoder model")

#END PHASE 2

#PHASE 3: chatbot

from keras.preprocessing.sequence import pad_sequences
print("##########################################")
print("#             start chatting             #")
print("##########################################")

'''
ENG: Create chat flow, we provide actual examples to show dec/enc sequence and text predictions
ITA: Creiamo il flusso della chat, dando esempi concreti per mostrare la sequenza enc/dec e le previsioni
'''
    
prepro1 = ""
while prepro1 != 'q':
    # 'q' used to quit the loop
    
    prepro1  = input("you : ")
    # prepro1 = "Hello"

    prepro1 = clean_text(prepro1)
    # prepro1 = "hello"

    prepro = [prepro1]
    # prepro1 = ["hello"]

    txt = []
    for x in prepro:
        # x = "hello"
        
        lst = []
        for y in x.split():
            # y = "hello"
            
            try:
                lst.append(vocab[y])
                # vocab['hello'] = 454
                
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)
        # txt = [[454]]
    
    '''
    ENG: Convert to numpy array of defined lenght to encode words and tokens
    ITA: convertiamo il testo in numpy array della lunghezza definita per codificare parole e tokens
    '''
    
    txt = pad_sequences(txt, 13, padding='post')
    # txt = [[454,0,0,0,.........13]]

    stat = enc_model.predict( txt )

    empty_target_seq = np.zeros( ( 1 , 1) )   
    #   empty_target_seq = [0]

    empty_target_seq[0, 0] = vocab['<SOS>']
    #    empty_target_seq = [255]
    
    '''
    ENG: Define End of sentence condition
    ITA: definiamo la condizione di fine frase
    '''
    
    stop_condition = False
    decoded_translation = ''

    while not stop_condition :
        
        '''
        ENG: Define output probility via decoder model (use argmax to maximize the single probability)
        ITA: definiamo le probabilità dell'output tramite il modello di deconding (usiamo argmax per massimizzare le singole probabilità)
        '''
        
        dec_outputs , h, c= dec_model.predict([ empty_target_seq] + stat )
        decoder_concat_input = dense(dec_outputs)
        # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

        sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
        # sampled_word_index = [2]
        
        '''
        ENG: Using reverse vocabulary to take back words
        ITA: usiamo il vocabolario inverso per riprendere le parole
        '''
        
        sampled_word = inv_vocab[sampled_word_index] + ' '

        # inv_vocab[2] = 'hi'
        # sampled_word = 'hi '
        
        '''
        ENG: Check if sampled word is not EOS token
        ITA: controlliamo se la parola non è un token di fine frase
        '''
        
        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word  

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 13:
            stop_condition = True 
        '''
        ENG: Variable resetting before passing to the next word
        ITA: resettiamo le variabili prima di passare alla prossima parola
        '''
        
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        # <SOS> - > hi
        # hi --> <EOS>
        stat = [h, c]  

    print("chatbot : ", decoded_translation )
    print("==============================================")  

#END PHASE 3













