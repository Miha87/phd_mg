'''REFERENCA
    Li, Shi-Jie, et al. "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation." 
    IEEE Transactions on Pattern Analysis and Machine Intelligence (2020).

Dodatan izvor:
    https://github.com/sj-li/MS-TCN2/blob/master/model.py
    
Moja implementacija je prijevod originalnog koda iz PyTorch-a u Tensorflow 2.0.

Implementacija pomoćnih modula napravljena je u subclassing stilu, a glavnog modela u funkcionalnom stilu.
Moji dodatci izvornom kodu su: 
    
    a)Izum i dizajn novog sloja "MaskConv1D" koji omogućuje propagiranje maske kroz 1d konvolucijski
    sloj, što nije dosad bilo podržano u TFlow-u, a potrebno je  ako želimo raditi 
    na efikasan način sa nizovima različitih duljina kroz dodavanje nadopuna. Bez 
    ovog sloja se mora raditi ili sa mini grupom veličine 1 (što je sporo), ili
    je potrebno prihvatiti nadopunu kao pozadinsku klasu i pustiti model da se nosi sa time što
    je rasipanje u vidu trening ciklusa potrebnih da se nauči kako izgleda pozadina.
    Autori članka su napravili previd, jer rade maskiranje tek kod funkcije gubitka,
    a potrebno je raditi maskiranje svakog sloja ako želimo dobiti jednak izlaz modela
    za isto opažanje neovisno od nadopune. Tj. u njihovoj implementacij prolaz unaprijed
    bi za isto opažanje, ovisno o tome da li je nadopunjeno, dao različite vrijednosti
    što nije korektno!!!
    
    b)Implementacija funkcije cilja prema prijedlogu iz članka za redukciju 
    prekomjerne segmentiranosti
    
    c)Mogućnost izbora između dijeljenih i zasebnih parametara u fazi rafiniranja predikcija,
    u izvornom kodu stavljeni su dijeljeni parametri. Međutim u članku je jasno
    pokazano da ušteda u broju parametara smanjuje učinkovitost modela, tako da sam definirao
    mogućnost izbor.
    
    d)Dokumentirao sam implementaciju prema objašnjenjima iz članka, 
    jer je izvorni kod nije imao.
    
NAPOMENA: Funkcija za izgradnju finalnog modela namjerno je implementirana 
funkcionalnim API-em zbog tehničkih benefita (npr. spremanje, podizanje, lakše debuggiranje itd.),
dok su slojevi implementirani subclassing pristupom, također zbog tehničkih benefita.
'''
#%%Biblioteke
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
import tensorflow as tf

#%%Klasa koja će omogućiti dodavanje maske izlazu iz Conv1D sloja
#Workaround koji omogućuje propagiranje maske do funkcije gubitka

class MaskConv1D(layers.Layer):
    '''Klasa za propagiranje maske kroz Conv1D slojeve.
    Prima izračunatu masku ulaznog opažanja i primjenjuje je na transformaciju
    iz Conv1D sloja.'''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Omogućava propagiranje maske
        self.supports_masking = True
        #Sa ovim dodatkom rješeno je pitanje izlazne maskirane vrijednosti
        self.masking = layers.Masking(mask_value=0.)
    
    #Omogućava prijem maske, kroz argument "mask"
    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        masked_inputs = inputs * broadcast_float_mask   
        masked_inputs = self.masking(masked_inputs)
        return masked_inputs
    
    #Omogućava spremanje i podizanja sloja sa pohranjenim argumentima konstruktora
    def get_config(self):
        base_config = super().get_config()
        return base_config

#%%Subclassing implementacija pomoćnih modula MS-TCN++ modela

#Dilatirani rezidualni modul
class DilatedResidualModule(layers.Layer):
    '''Kod ovog modula cilj primjene dilatacije je povećanje receptivnog polja,
    uz održavanje jednakog broja parametara. Preskočna veza olakšava učenje. Nije
    moguće samostalno korištenje ovog modula jer će dimenzije transformacijske i preskočne
    grane biti različite!!!'''
    
    #Definicija elemenata modula
    def __init__(self, filters, dilation_rate, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        ##Definiranje hiperparametara modula
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        ##Definicija slojeva u modulu
        #Slojevi maskiranja
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D()
        
        #Ova komponenta kroz stopu dilatacije omogućava veće receptivno polje
        #uz isti broj parametara, bitno zbog hvatanja vremenskog konteksta u opažanjima
        self.conv_dilated = layers.Conv1D(filters, kernel_size=3, padding="same",
                                          dilation_rate=dilation_rate,
                                          activation="relu")
        
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        
        self.dropout = layers.Dropout(dropout_rate)
    
    #Prolaz unaprijed kroz sloj
    def call(self, inputs):
        #Izračun ulazne maske
        mask = self.masking.compute_mask(inputs)
        
        #Maskirana preskočna veza
        masked_inputs = self.masking(inputs)
        
        #Prolaz kroz dilatirani konv. sloj
        x = self.conv_dilated(masked_inputs)
        x = self.conv_mask(x, mask=mask)
        
        #Prolaz kroz konv.
        x = self.conv_1x1(x)
        x = self.conv_mask(x, mask=mask)
        
        #Regularizacija
        x = self.dropout(x)
        
        #Zbrajanje preskočne i transformacijske grane
        output = layers.add([masked_inputs, x])
        
        return output
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitna kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, 
                "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate}

#Modul za generiranje inicijalnih predikcija
class PredictionGeneration(layers.Layer):
    '''Služi za generiranje inicijalnih predikcija. Ulaz ovog modula su značajke 
    svake sličice videa. Ključna komponenta je dualni dilatirani sloj, gdje jedna 
    komponenta inicijalno ima veliko receptivno polje te ga smanjuje kroz slojeve, 
    dok druga komponenta ima malo receptivno polje te ga kroz slojeve povećava. 
    Razlog ovog dizajna je da želimo uhvatiti lokalni i globalni temporalni 
    kontekste opažanja. Također prednost ovog pristupa je da je sačuvana 
    temporalna rezolucija ulaza.'''
    
    #Definicija elemenata modula
    def __init__(self, num_layers, filters, num_classes, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        ##Definiranje hiperparametara modula
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        ##Definicija slojeva u modulu
        #Slojevi maskiranja
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D()
        
        #Sloj podešavanje dimenzionalnosti ulaza
        self.conv_in = layers.Conv1D(filters, kernel_size=1)
        
        #Dualni dilatirani sloje
        self.conv_dilated_1 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**(num_layers - 1 - i)) for i in range(num_layers)]
        
        self.conv_dilated_2 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**i) for i in range(num_layers)]
        
        #Sloj spajanja dualnih komponenti
        self.conv_fusion = [layers.Conv1D(filters, kernel_size=1, activation="relu") for i in range(num_layers)]
        
        self.dropout = layers.Dropout(dropout_rate)
        
        #Sloj izračuna predikcije modula
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")
    
    #Prolaz unaprijed kroz sloj
    def call(self, inputs):
        #Izračun ulazne maske
        mask = self.masking.compute_mask(inputs)
        
        #Podešavanje dimenzionalnosti ulaza
        x = self.conv_in(inputs)
        x = self.masking(x)
        
        #Petlja kroz slojeve
        for i in range(self.num_layers):
            #Preskočna veza
            shortcut = x
            
            #Dilatacija čije se receptivno polje smanjuje kroz slojeve
            dilated_1 = self.conv_dilated_1[i](x)
            dilated_1 = self.conv_mask(dilated_1, mask=mask)
            
            #Dilatacija čije se receptivno polje povećava kroz slojeve
            dilated_2 = self.conv_dilated_2[i](x)
            dilated_2 = self.conv_mask(dilated_2, mask=mask)
            
            #Spajanje dilatiranih grana
            concat = layers.concatenate([dilated_1, dilated_2], axis=-1)
            fusion = self.conv_fusion[i](concat)
            fusion = self.conv_mask(fusion, mask=mask)
            
            #Regularizacija
            x = self.dropout(fusion)
            #Zbrajanje transformacijske i preskočne grane
            x = layers.add([x, shortcut])
        
        #Računanje izlazne predikcije iz modula
        output = self.conv_out(x)
        output = self.conv_mask(output, mask=mask)
        
        return output
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitna kod podizanja/spremanja modela  
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate}

#Modul za rafiniranje predikcija
class Refeinment(layers.Layer):
    '''Služi za podešavanje inicijalnih predikcija kroz etape. Svaka etapa ovog 
   modula prima predikciju prethodne etape te je dodatno rafinira. Kako je ulaz svake 
   etape predikcija prethodne to omogućava učenje dobrih nizova aktivnosti što 
   smanjuje prekomjernu segmentiranost. Pokazano je da je bitno da je ulaz svake 
   etape samo niz predikcija, a ne niz predikcija i dodatne značajke. To loše 
   utječe na model, zato jer su večinom sve aktivnosti vizualno slične pa to buni 
   model i navodi ga na prekomjernu segmentaciju.'''
   
    #Definicija elemenata modula
    def __init__(self, num_layers, filters, num_classes, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        ##Definiranje hiperparametara modula
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        ##Definicija slojeva u modulu
        #Slojevi maskiranja
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D()
        
        #Sloj podešavanje dimenzionalnosti ulaza
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        #U originalnoj implementaciji oni koriste dijeljene težine tj. isti modul više
        #puta, time smanjuju broj parametara, ali gube na točnosti modela
        self.dilated_residual_blocks = [DilatedResidualModule(filters, dilation_rate=2**i, dropout_rate=dropout_rate) 
                                        for i in range(num_layers)]
        
        #Sloj izlazne predikcije modula
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")
        
    #Prolaz unaprijed kroz sloj
    def call(self, inputs):
        #Izračun ulazne maske
        mask = self.masking.compute_mask(inputs)
        
        #Podešavanje dimenzionalnosti ulaza
        x = self.conv_1x1(inputs)
        x = self.masking(x)
        
        #Petlja po rezidualnim modulima
        for block in self.dilated_residual_blocks:
            x = block(x)
        
        #Računanje izlazne predikcije iz modula
        output = self.conv_out(x)
        output = self.conv_mask(output, mask=mask)
        
        return output
        
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitna kod podizanja/spremanja modela  
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate}
    
#Funkcija za izgradnju više etapnog temporalnog konvolucijskog modela
def build_ms_tcn_model(input_shape=(None,2048), training=True, mask_value=0., num_layers_PG=11, num_layers_R=10, 
                    R_stages=3, shared_R=False, filters=64, num_classes=10, dropout_rate=0.5, seed=42):
    """
    Funkcija za izgradnju više etapnog temporalnog konvolucijskog modela (MS-TCN++).
    Za razliku od drugih modela koji zahtjevaju poduzorkovane nizove niske 
    temporalne rezolucije (npr. 1 - 3 fps), ovaj model se pokazao učinkovitim 
    na visokoj rezoluciji (npr. 15 fps), ali ništa manje uspješan nije na 
    nizovima niske rezolucije.

    Argumenti:
    ----------
    input_shape: tuple
        Dimenzionalnost ulaznog vremenskog niza. U slučaju da se koriste nizovi
        različitih duljina prva dimenzija je None. Preddefinirana vrijednost je 
        (None, 2048).
    
    training: bool
        Da li model radi u režimu za učenje ili evaluaciju. Ako je u režimu učenja
        tada model ima izlaznu granu iz svake etape zbog brže konvergencije. Kod 
        evaluacije model ima izlaz samo iz posljednje etape. Preddefinirana 
        vrijednost je True.
        
    mask_value: int/float
        Vrijednost nadopune vremenskog niza koju je potrebno zanemariti kod
        kalkulacija. Mora odgovarati tipu podataka u vremenskom nizu. 
        Preddefinirana vrijednost je 0..
              
    num_layers_PG: int
        Broj slojeva u modulu za generiranje inicijalnih predikcija. 
        Preddefinirana vrijednost je 11.
        
    num_layers_R: int
        Broj slojeva u svakom pojedinačnom modulu za rafiniranje predikcija. 
        Preddefinirana vrijednost je 10.
        
    R_stages: int
        Broj etapa rafiniranja predikcija. Ovo odgovara broju modula za rafiniranje 
        predikcija. Preddefinirana vrijednost je 3.
        
    shared_R: bool
        Da li se parametri modula za rafiniranje predikcija dijele između etapa, tj.
        da li svi moduli koriste iste težine. 
        Smanjuje broj parametara, ali i točnost modela. .Preddefinirana vrijednost je False.
        
    filters: int
        Broj filtera u svakom konvolucijskom sloju modela. Preddefinirana vrijednost je 64.
        
    num_classes: int
        Broj klasa za koji učimo model, odgovara broju filtera u zadnjem konvolucijskom
        sloju. Preddefinirana vrijednost je 10.
    
    dropout_rate: float [0, 1.]
        Stopa ulaznih neurona koje treba nasumično izbaciti sa svrhom regularizacije.
    
    seed: int
        Za reprodukciju rezultata.

    Povratna vrijednost:
    --------------------
    model: Model
        Instanca Model klase.
    """ 
    K.clear_session()
    
    tf.random.set_seed(seed)
    
    ##Definicija elementa modela
    inputs = layers.Input(shape=input_shape)
    masking = layers.Masking(mask_value=mask_value)
    PG = PredictionGeneration(num_layers_PG, filters, num_classes, dropout_rate)
    
    #Da li želimo dijeljene parametre etapa rafiniranja predikcija
    if shared_R:
        shared_refeinment = Refeinment(num_layers_R, filters, num_classes, dropout_rate)
        Rs = [shared_refeinment for _ in range(R_stages)]
    else:
        Rs = [Refeinment(num_layers_R, filters, num_classes, dropout_rate) for _ in range(R_stages)]
    
    ##Prolaz unaprijed 
    #Maskiranje nadopuna 
    x = masking(inputs)
    
    #Generiranje ulaznih predikcija
    x = PG(x)
    
    #Model ima izlaz za svaku etapu (npr. 4 etape, 4 izlaza). Kod evaluacije se gleda 
    #predikcija završne etape, ostali izlazi pomažu bržoj konvergenciji rješenju
    if training:
        outputs = [x]
    
        #Petlja po etapama rafiniranja predikcija
        for R in Rs:
            x = R(x)
            #Dodavanje izlaza
            outputs.append(x)
    else:
        #Petlja po etapama rafiniranja predikcija
        for R in Rs:
            x = R(x)
        
        outputs = x
    
    return Model(inputs=inputs, outputs=outputs)
  
#%%Funkcija gubitka za redukciju prekomjerne segmentacije
class SegmentationLoss(tf.keras.losses.Loss):
    '''Gubitak za problem video segmentacije. Sastoji se od kombinacije
    dva gubitka: klasične unakrsne entropije za problem klasifikacije te 
    srednjeg kvadrata odstupanja između log vjerojatnosti dva uzastopna vremenska
    koraka koji je ograničen (eng. "truncated") sa gornjom granicom (tau). Zadatak
    druge komponente gubitka je smanjiti prekomjernu segmentaciju tj. izgladiti gubitak.
    
    Funkcija podržava stvarne oznake PRIJE pretvorbe u one-hot kodirane oznake!!!
    
    Argumenti:
    ----------
    lmbda: float
        Udio gubitka izglađivanja u cijeloj funkciji gubitka. 
        Preddefinirana vrijednost je 0.15.
    
    tau: float
        Gornja granica pojedinačnih vrijednosti gubitka izglađivanje, tj.
        ako je apsolutna vrijednost razlike u log vjerojatnostima između trenutnog 
        i prethodnog vremenskog koraka veća od ove vrijednosti, biti će reducirana na vrijednost
        definiranu sa tau. N.B može smanjiti točnost modela ako je vrijednost prevelika
        , jer penalizira  situacije u kojima je model jako siguran da je došlo do 
        promjene klase između susjednih vremenskih koraka, a gubitak ga odvraća od takve odluke.
        Preddefinirana vrijednost je 4..
    
    num_classes: int
        Broj klasa za koji učimo model. Preddefinirana vrijednost je 10.
    
    Povratna vrijednost:
    --------------------
    loss: Loss
        Instanca tf.keras.losses.Loss klase.
    '''
    
    #Definiranje hiperparametara gubitka
    def __init__(self, lmbda=0.15, tau=4., num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.lmbda = lmbda
        self.tau = tau
        self.num_classes = num_classes
        #Obavezno bez redukcije(SUM ili slično) nju radi automatski framework
        self.MSE_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.CCE_loss = SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
          
    #Izračun gubitka
    def call(self, y_true, y_pred):
               
        ##CCE gubitak## - gubitak klasifikacije
        CCE_loss = self.CCE_loss(y_true, y_pred)
        
        ##T-MSE gubitak## - gubitak izglađivanja
        #Log vjerojatnosti trenutnog i prethodnog vremenskog koraka
        #Konkatencija je WORKaround jer gubitak zahtjeva istu dimenzionalnost ulaza i izlaza pa dodajemo zadnji vremenski koraka
        y_pred_t1 = tf.math.log_softmax(tf.concat([y_pred[:, 1: ], tf.stop_gradient(y_pred[:, -1:])], axis=1), axis=-1) 
        y_pred_t0 = tf.math.log_softmax(tf.concat([y_pred[:, :-1], y_pred[:, -1:]], axis=1), axis=-1)
        
        #Gradijent računamo samo s obzirom na trenutni vremenski korak, prethodni smatramo konstantom
        #Režemo vrijednosti u raspon [0, tau]
        T_MSE_loss = tf.clip_by_value(self.MSE_loss(y_pred_t1, tf.stop_gradient(y_pred_t0)), 0., tf.square(self.tau))
        
        return  tf.add(CCE_loss, tf.multiply(self.lmbda, T_MSE_loss))
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitno kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "lmbda": self.lmbda, "tau": self.tau,
                "num_classes": self.num_classes}