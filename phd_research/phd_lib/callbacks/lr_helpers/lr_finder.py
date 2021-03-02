#REFERENCA:
'''
    Smith, Leslie N. "Cyclical learning rates for training neural networks." 
    2017 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2017.
    https://arxiv.org/abs/1506.01186
'''

#PRIMJER:
'''python

    lr_finder = LearningRateFinder(model, weight_file, stop_factor, smoothing_factor)
    lr_finder.find_lr(X, y, num_samples, batch_size, epochs, min_lr, max_lr)
    lr_finder.plot_loss_vs_lrs(smooth_losses)
    
'''   
#%%Biblioteke
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

#%%Klasa za pronalaženje optimalne donje i gornje granice stope učenja

class LearningRateFinder:
    '''Klasa za pronalaženje optimalne donje i gornje granice stope učenja.'''
    
    def __init__(self, model, weights_file=None, stop_factor=4, smoothing_factor=None):
        '''Sprema kompajlirani model, putanju datoteke u kojoj pohranjujemo početne
        vrijednosti težina modela, faktor zaustavljanja učenja u slučaju velike 
        vrijednosti funkcije gubitka, faktor izglađivanja vrijednosti
        funkcije gubitka. Ako je definiran, faktor izglađivanja mora biti u 
        rasponu [0, 1>.
        
        U slučaju da se koristit random seed kod inicijalizacije modela, nije
        potrebno spremati težine modela, već nakon pronalaska optimalne stope učenja
        ponovno inicijalizirati model sa istim seedom.'''
        
        self.model = model
        self.weights_file = weights_file
        self.stop_factor = stop_factor
        if smoothing_factor is not None and smoothing_factor >= 1.:
            raise ValueError("Faktor izglađivanja mora biti u rasponu [0,1>")
        self.smoothing_factor = smoothing_factor
       
        #Stopa učenja po iteracijama
        self.rates = []
        #Oznaka trenutne iteracije, potrebna je za korekciju EWMA gubitka
        #ne možemo koristi broj batch-a koji vraća Keras, jer on uvijek vraća
        #broj batcha u epohi, što bi uzrokovalo pogrešnu kalkulaciju korekcije
        self.iter_num = 1
        #Sirovi gubitci
        self.raw_losses = []
        #Bilježenje najmanjeg ostvarenog sirovog gubitka, multiplicira
        #se sa faktorm zaustavljanja kako bi prekinuli učenje kada gubitak počne 
        #rasti uslijed prevelike stope učenja, bez ovoga vizualizacije optimalnih
        #granica najčešće nije moguća jer će neki od gubitaka biti previsok
        #da bi trendovi u grafu bili vidljivi!!
        self.lowest_loss = 1e9
        #Trenutne vrijednosti izglađenog gubitka, bez korekcije
        self.ewma_loss = 0
        #Korigirani ewma gubitaka za svaku iteraciju
        self.unbiased_ewma_losses = []
    
    def _reset_state(self):
        '''Privatna metoda za re-inicijalizaciju stanja instance klase, kako
        bi instancu mogli koristiti za više različitih podešenja granica.'''
        
        self.rates = []
        self.iter_num = 1
        self.raw_losses = []
        self.lowest_loss = 1e9
        self.ewma_loss = 0
        self.unbiased_ewma_losses = []
                    
    @staticmethod
    def clc_factor(min_lr, max_lr, iterations):
        '''Metoda za izračun multiplikativnog faktora povećanja stope učenja u svakoj iteraciji.'''
        return np.exp(np.log(max_lr / min_lr) / iterations)
    
    @staticmethod
    def loss_smoother(batch_loss, ewma_loss, smoothing_factor, iter_num):
        '''Metoda za izglađivanje vrijednosti funkcije gubitka primjenom
        EWMA algoritma.'''
        
        #Trenutna vrijednost eksponencijalno otežanog pomičnog prosjeka bez korekcije pristranosti
        updated_ewma_loss = smoothing_factor * ewma_loss + (1 - smoothing_factor) * batch_loss
        #Korigirana vrijednost EWMA- korekcija je izrazito potrebna kod početnih iteracija kada 
        #većina doprinosa u prosjeku dolazi od batch_loss-a
        unbiased_ewma_loss= updated_ewma_loss / (1 - (smoothing_factor ** iter_num))
        
        return updated_ewma_loss, unbiased_ewma_loss
    
    def _on_batch_end(self, batch, logs=None):
        '''Privatna metoda koja će biti korištena u okviru callback-a. 
        Na kraju treniranja svake mini grupe, Keras će joj automatski 
        prosljediti logs i batch parametar.'''
        
        #Spremanje gubitka za trenutnu mini grupu
        batch_loss = logs["loss"]
        self.raw_losses.append(batch_loss)
        #Spremanje stope učenja korištene u završenoj iteraciji,
        lr = K.get_value(self.model.optimizer.lr)
        self.rates.append(lr)
    
        #U slučaju da je definiran faktor izglađivanja izračunaj i izglađene gubitke
        if self.smoothing_factor is not None:
            self.ewma_loss, unbiased_ewma_loss = self.loss_smoother(batch_loss, self.ewma_loss, 
                                                          self.smoothing_factor, self.iter_num)
            self.unbiased_ewma_losses.append(unbiased_ewma_loss)
        
        #Gornja granica gubitka, ako je trenutni gubitak iznad te granice i napravili smo barem 2 iteracije
        #zaustavi učenje
        stop_loss = self.lowest_loss * self.stop_factor
        if batch_loss > stop_loss and self.iter_num > 1:
            print("[INFO] gubitak je veći od izračunate gornje granice, prekidam učenje")
            self.model.stop_training=True
            return
            
        #Ažuriranje najmanjeg ostvarenog gubitka u procesu učenja
        if batch_loss < self.lowest_loss:
            self.lowest_loss = batch_loss
        #Ažuriranje stope učenja za slijedeću iteraciju
        updated_lr = lr * self.factor
        K.set_value(self.model.optimizer.lr, updated_lr)
        #Ažuriranje broja iteracije
        self.iter_num += 1
         
    def find_lr(self, X, y, num_samples, batch_size, epochs=5, 
                min_lr=1e-6, max_lr=10):
        '''Metoda za traženje optimalnih granica stope učenja. Potrebno je definirati
        minimalnu i maksimalnu stopu učenja unutar čijih granica se kreće stopa učenja.
        Kao ulaz može primiti podatke koji stanu u memoriju, ali i one
        koji se obrađuju online. y=None, ako koristimo tf.data.dataset. 
        Potrebno definirati broj opažanja, te veličinu mini grupe i broj epoha.'''
        
        #Čišćenje postojećeg stanja
        self._reset_state()
        
        if self.weights_file is not None:
            print("[INFO] spremam početne vrijednosti težina i inicijalnu stopu učenja...")
            self.init_lr = K.get_value(self.model.optimizer.lr)
            self.model.save_weights(self.weights_file)

        print("[INFO] tražim optimalne granice stope učenja...")
        #Izračun broja iteracija
        iterations = np.ceil(num_samples / batch_size) * epochs
        #Izračun faktora povećanja stope učenja
        self.factor = self.clc_factor(min_lr, max_lr, iterations)
        #Postavljanje stope učenja na definiranu minimalnu vrijednost stope učenja
        K.set_value(self.model.optimizer.lr, min_lr)
        #Definiranje callbacka za bilježenje metrike po završetku mini grupe,
        #podešavanje stope učenja, i zaustavljanje u slučaju prevelikog rasta f.gubitka
        callback = LambdaCallback(on_batch_end=lambda batch, logs: 
                                  self._on_batch_end(batch, logs))
        
        #U slučaju da je ulaz tf.data.dataset
        if y is None:
            self.model.fit(x=X, y=None, epochs=epochs, callbacks=[callback])
        else:
            self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=[callback])
        
        print("[INFO] postupak traženja optimalnih granica stope učenja završen")
        print("[INFO] generirajte plot i odaberite optimalne granice prije novog ciklusa učenja")
        
        if self.weights_file is None:
            print("[INFO] ponovno inicijalizirajte model sa istim seed-om") 
        else:
            print("[INFO] vraćam početne težine modela i inicijalnu stopu učenja")
            K.set_value(self.model.optimizer.lr, self.init_lr)
            self.model.load_weights(self.weights_file)
            
    def plot_loss_vs_lrs(self, smooth_losses=False):
        '''Generiranje plota gubitka kao funkcije stope učenja,
        može prikazati sirove ili izglađene gubitke ako su izračunti.'''
        
        rates = self.rates
        if self.smoothing_factor is None:
            raise ValueError("Izglađeni gubitci nisu izračunati,\
                             mogu prikazati samo sirove gubitke")
        if smooth_losses:
            losses = self.unbiased_ewma_losses
        else:
            losses = self.raw_losses
        #Za prikaz stope učenja koristim logaritamsku skalu
        plt.plot(rates, losses)
        plt.xscale("log")
        plt.xlabel("Learning rate (Log)")
        plt.ylabel("Loss")