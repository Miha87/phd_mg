#REFERENCA:
'''
    Smith, Leslie N. "Cyclical learning rates for training neural networks." 
    2017 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2017.
    https://arxiv.org/abs/1506.01186
'''
#PRIMJER:
'''python

    clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular')
    model.fit(..., callbacks=[clr])
    
'''  
#%%Biblioteke
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np

#%%Klasa za generiranje cikličke strategije stope učenja
class CyclicLR(Callback):
    """Ovaj callback implementira cikličku strategiju upravljanja stopom
    učenja. Ova klasa ciklički mjenja stopu učenja između dvije granice konstantnom
    frekvencijom. Amplituda ciklusa može biti skalirana na bazi informacije o 
    iteraciji ili ciklusu.
    
    Klasa sadrži tri ugrađene strategije:    
    "triangular":
        Osnovni trokutasti ciklus bez skaliranja amplitude.
    "triangular2":
        Osnovni trokutasti ciklus uz skaliranje veličine amplitude za pola
        u svakom ciklusu.
    "exp_range":
        U ovom slučaju inicijalna amplituda skalirana je gamma**(cycle iterations)
        faktorom u svakom ciklusu. 
     
    # Primjer 1:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Primjer 2:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    Argumenti
    ---------
    base_lr: float
        Inicijalna stopa učenja koja je donja granica u ciklusu
            
    max_lr: float
        Gornja granica ciklusa. Na temelju nje se funkcijom definira amplituda
        ciklusa (max_lr - base_lr). Stopa učenja u bilo kojem ciklusu je
        suma base_lr i nekog skaliranja amplitude; stoga max_lr i ne mora
        nužno biti dostignut ovisno o primjenjenoj funkciji skaliranja.
    
    step_size: float/int
        Broj trening iteracija u polu ciklusu. Autor članka predlaže korištenje
        step_size koji je 2-8 x broj trening iteracija u epohi.
    
    mode: str => jedna od {triangular, triangular2, exp_range}.
        Preddefinirana vrijednost je 'triangular'.
        Vrijednost odgovara odabranoj strategiji cikličkog ponašanja. Ako je
        parametar scale_fn različit od None, ovaj argument se zanemaruje!
    
    gamma: float/int
        Konstanta u 'exp_range' funkciji skaliranja: gamma**(cycle iterations).
        Preddefinirana vrijednost je 1..
    scale_fn: fun
        Proizvoljna strategija skaliranja definirana sa lambda funkcijom koja
        prima jedan argument (x), pri čemu mora vrijediti:
            0 <= scale_fn(x) <= 1 za svaki x >= 0. 
        Ako je definiran argument mode je zanemaren.
    
    scale_mode: str => jedan od  {'cycle', 'iterations'}
        Definira da li je scale_fn izračunata na temelju broja ciklusa ili iteracije.
        Preddefinirana vrijednost je 'cycle'.
            
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        '''Spremanje inicijalne stope učenja, maksimalne stope učenja, broja trening
        iteracija u polu ciklusu, strategije cikličkog ponašanja, faktora gamma,
        funkcije skaliranja i moda izračuna funkcije skaliranja'''

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        #Da li je defnirana funkcija skaliranja amplitude ciklusa
        if scale_fn == None:
            if self.mode == 'triangular':
                #Nema skaliranja amplitude ciklusa stope učenja
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                #Amplituda (tj. max. stopa učenja) je u svakom ciklus upola manja
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                #Amplituda je u svakoj iteraciji skalirana sa gamma ** (x)
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            #Ako postoji funkcija skaliranja onda je i definiran način na koji
            #se ona primjenjuje na amplitudu ciklusa
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        #Inicijalizacija broja iteracija u ciklusu, broja iteracija učenja
        #objekta za čuvanje informacija o metrici, ali i o stopi učenja i iteraciji
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        
        #Kod inicijalizacije (početak učenja), postavi broj iteracija u ciklusu
        #na 0
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Reset svih postavki."""
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        '''Izračun stope učenja na temelju ciklusa i broja iteracija u ciklusu.'''
        
        #U kojem smo ciklusu (broj iteracija / (2 * broj iteracija u polu ciklusu))
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        #Na ovaj način osiguravamo da je x uvijek u rasponu [-1,1]
        #Tj. max. vrijednost stope učenja (amplitudu), ostvarujemo u 0
        #Radimo pomak ciklusa u 0
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            #Ništa drugo nego jednadžba pravca, a x je pozicija na x osi, 
            #1-x, kako bi x odgovarao stvarnom x za koji nas zanima stopa učenja
            #npr iz gornjeg izraza za x koji je odgovarao 0.8 poziciji dobivamo
            #-0.2, a za onaj koji je bio 1.2 dobivamo 0.2, prije primjene apsolutne vrj.
            #oba odgovaraju istoj stopi učenja koja mora biti vezana uz poziciju 0.8 
            #skaliranje mijenja nagib pravca, i događa se po ciklusu
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            #skaliranje se događa u svakoj iteraciji
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        '''Metoda koja definira ponašanje klasa na početku treninga,
        u trenutnoj Keras implementaciji logs argument ne prima ništa.'''
        logs = logs or {}
        
        #Ako je broj iteracija u okviru ciklusa 0, stopa učenja se postavlja
        #na donju granicu, inače se za izračun koristi funkcija clr
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_train_batch_end(self, batch, logs=None):
        '''Metoda koja definira ponašanje nakon obrade mini grupe opažanja,
        logs argument dobiva objekt sa metrikom izračunatom na mini grupi'''
        
        logs = logs or {}
        #Po obradi mini grupe ažuriraj broj trening i ciklus iteracija
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        #u history objekt dodaj informaciju o trenutnoj stopi učenja i broju trening iteracija
        #ako postoji ključ vrati objekt pod tim ključem, ako ne dodaj ga sa default vrijednosti ([])
        #i vrati taj objekt =>setdefault
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        #Dodaj vrijednosti iz trenutne epohe u postojeći history objekt
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        #Izračunaj novu stopu učenja i dodaj je u optimizacijski postupak
        K.set_value(self.model.optimizer.lr, self.clr())