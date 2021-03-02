#REFERENCA
'''
GLAVNI DIO IMPLEMENTACIJE:
  Adrian Rosebrock, "Pyimagesearch" In: https://www.pyimagesearch.com 
  
  NAPOMENA: moji dodatci su praćenje omjera val/train gubitka, i funkcija 
  monitor_data_reader
'''
#PRIMJER:
'''python

    tm = TrainingMonitor(plot_path, history_path, start_at_epoch)
    model.fit(..., callbacks=[tm])
    
'''    
#%%Biblioteke
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pandas as pd

#%%Klasa za nadzor procesa učenja 

class TrainingMonitor(BaseLogger):
    '''Klasa za nadzor procesa učenja po epohi'''
    
    def __init__(self, plot_path, history_path=None, start_at_epoch=0):
        '''Sprema putanju lokacije plota gubitka (png) i definiranih metrika. 
        Sprema putanju do JSON serijalizirane datoteke sa vrijednostima
        gubitka i metrika iz Keras history objekta ako želimo rekreirati 
        cijelu povijest od početne epohe te ih sve plotati.
        Sprema epohu od koje model kreće/nastavlja učenje, bitno za plotanje.
        '''
        super(TrainingMonitor, self).__init__()
        self.plot_path = plot_path
        self.history_path = history_path
        self.start_at_epoch = start_at_epoch
    
    def on_train_begin(self, logs={}):
        '''Ovu metodu Keras automatski poziva samo JEDNOM kada započne proces učenja.
        NB logs je parametar kroz koji Keras šalje stvari zapisane na početku
        učenja, ali za ovu metodu taj argument u trenutnoj Keras implementaciji 
        ne prima nikakve argumente.'''
        
        #Incijalizacija history riječnika
        self.H = {}
        #Ako postoji serijaliziran history objekt
        if self.history_path is not None:
            if os.path.exists(self.history_path):
                #Otvori dokument, deserijaliziraj ga
                self.H = json.loads(open(self.history_path).read())
                #Provjeri od koje epohe nastavljamo učenje
                if self.start_at_epoch > 0:
                    #Ako history sadrži i logove za kasnije epohe
                    #odbacujemo ih, te krećemo od "start_at_epoch" epohe
                    for k in self.H.keys():
                        #svaki ključ mapira listu
                        self.H[k] = self.H[k][:self.start_at_epoch]
                        
    def on_epoch_end(self, epoch, logs={}):
        '''Što se radi na kraju epohe treniranja. Kroz logs Keras prosljeđuje
        metriku i gubitke za trenutnu epohu, a kroz epoch broj epohe 
        (ne moramo nužno koristiti ni jedan od tih argumenta)'''
        
        #Custom metrika za dodati u history objekt
        #Omjer validacijskog i trening gubitka - za praćenje pre/podnaučenosti
        losses_ratio = float(logs["val_loss"] / logs["loss"])
        self.H.setdefault('val_loss/loss', []).append(losses_ratio)
        
        #Petlja po elementima logs objekta koji ima iste ključeve kao i history objekt
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            #na postojeću listu u history objektu dodajemo nove elemente iz logs objekta
            #poziv funkciji float je, zato jer se ne može serijalizirati float32 objekt u json
            l.append(float(v))
            #u history objekt spremamo ažuriranu metriku sa dodanim podatcima iz logs-a
            self.H[k] = l
        
        #Ako je definirana putanja za spremanja history objekta u JSON formatu
        if self.history_path is not None:
            with open(self.history_path, "w") as f:
                f.write(json.dumps(self.H))
        
        #Ako su u history objektu spremljene barem 2 epohe učenja
        if len(self.H["loss"]) > 1:
            #Napravimo plot trening i validacijskih gubitaka i točnosti
            #Ostale metrike ne plot-amo jer su značajno različitih mjernih jedinica
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title(f"Train and Val loss and accuracy [Epoch {len(self.H['loss'])}]")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            #Spremanje trenutne slike, ovo treba podesiti
            plt.savefig(self.plot_path)
            plt.close()

#%%Specijalna klasa za moje modele sa 2 izlaza - zbog ključeva u logs riječniku radim ovu podklasu

class MultiOutTrainingMonitor(TrainingMonitor):
    '''Klasa za nadzor procesa učenja kod modela sa 2 izlaza sa specifičnim nazivima. Jedina
    izmjena u odnosu na TrainingMonitor klasu je metoda on_epoch_end.'''
    
    def on_epoch_end(self, epoch, logs={}):
        '''Što se radi na kraju epohe treniranja. Kroz logs Keras prosljeđuje
        metriku i gubitke za trenutnu epohu, a kroz epoch broj epohe 
        (ne moramo nužno koristiti ni jedan od tih argumenta)'''
        
        #Custom metrika za dodati u history objekt
        #Omjer validacijskog i trening gubitka - za praćenje pre/podnaučenosti
        #Zanima nas omjer vezan za zadnji izlaz jer će se na temelju njega raditi predikcija
        losses_ratio = float(logs["val_refeinment_loss"] / logs["refeinment_loss"])
        self.H.setdefault('val_loss/loss', []).append(losses_ratio)
        
        #Petlja po elementima logs objekta koji ima iste ključeve kao i history objekt
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            #na postojeću listu u history objektu dodajemo nove elemente iz logs objekta
            #poziv funkciji float je, zato jer se ne može serijalizirati float32 objekt u json
            l.append(float(v))
            #u history objekt spremamo ažuriranu metriku sa dodanim podatcima iz logs-a
            self.H[k] = l
        
        #Ako je definirana putanja za spremanja history objekta u JSON formatu
        if self.history_path is not None:
            with open(self.history_path, "w") as f:
                f.write(json.dumps(self.H))
        
        #Ako su u history objektu spremljene barem 2 epohe učenja
        if len(self.H["loss"]) > 1:
            #Napravimo plot trening i validacijskih gubitaka i točnosti
            #Ostale metrike ne plot-amo jer su značajno različitih mjernih jedinica
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["refeinment_loss"], label="train_loss")
            plt.plot(N, self.H["val_refeinment_loss"], label="val_loss")
            plt.plot(N, self.H["refeinment_accuracy"], label="train_acc")
            plt.plot(N, self.H["val_refeinment_accuracy"], label="val_acc")
            plt.title(f"Train and Val loss and accuracy [Epoch {len(self.H['loss'])}]")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            #Spremanje trenutne slike, ovo treba podesiti
            plt.savefig(self.plot_path)
            plt.close()

#%%Helper funkcija za čitanje logiranih history podataka

def monitor_data_reader(history_path):
    '''Pomoćna funkcija za čitanje podataka zapisanih 
    od strane trening monitora u JSON formatu.'''
    with open(history_path, "r") as f:
        history_data = json.loads(f.read())
        history_data = pd.DataFrame(history_data)
        return history_data
    