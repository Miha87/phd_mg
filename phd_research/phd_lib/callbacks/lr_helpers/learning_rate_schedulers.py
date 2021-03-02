#PRIMJER:
'''python

    schedule = StepDecay(init_lr, factor, drop_every)
    model.fit(..., callbacks=[LearningRateScheduler(schedule)])
    
'''   
#NAPOMENA: u verzijama tf.keras > 2.1, schedule funkcija mora kao ulaz
#primiti i inicijalnu stopu učenja!!!

#%%Biblioteke
import matplotlib.pyplot as plt
import numpy as np

#%%Bazna klasa rasporeda učenja

class LearningRateDecay:
    '''Bazna klasa za generiranje rasporeda stope učenja'''
    
    def plot(self, epochs, title="Learning Rate Schedule"):
        '''Metoda za crtanje stope učenja kao funkcije broja epoha'''
        
        #Izračun skupa stopa učenja za svaku epohu, naš objekt 
        #će biti callable (funkcija, koji prima informaciju o trenutnoj epohi
        lrs = [self(i) for i in epochs]
        
        #Plot rasporeda stope učenja
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
       
#%%Step decay LRS 
class StepDecay(LearningRateDecay):
    '''Klasa za generiranja koračnog LRS'''
    
    def __init__(self, init_lr=0.01, factor=0.25, drop_every=10):
        '''Inicijalizacija početne stope učenja, faktora za koji smanjujemo
        stopu učenja nakon definiranog broja epoha'''
        
        self.init_lr = init_lr
        self.factor = factor
        self.drop_every = drop_every
        
    def __call__(self, epoch):
        '''Izračun stope učenja za trenutnu epohu, ovo je metoda
        preko koje primamo informaciju u kojoj se trenutno epohi nalazimo'''
        
        #Kada broj epoha dostigne "drop_Every" faktor mjenja se faktor smanjenje
        #za ovu potenciju
        exp = np.floor((1 + epoch) / self.drop_every)     
        lr = self.init_lr * (self.factor ** exp)
        
        return float(lr)
   
#%%Linearni i polinomni LRS
class PolynomialDecay(LearningRateDecay):
    '''Klasa za generiranje polinomijalnog LRS'''
    
    def __init__(self, max_epochs=100, init_lr=0.01, power=1.0):
        '''Inicijalizacija najvećeg broja epoha prije nego stopa učenja postane
        0, početne stope učenja, potencija polinomne funkcije'''
        
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.power = power
        
    def __call__(self, epoch):
        '''Izračun stope učenja na temelju polinomne funkcije'''
        
        #Manja potencija funkcije rezultira sporijim smanjenjem stope učenja
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        lr = self.init_lr * decay
        
        return float(lr)
   