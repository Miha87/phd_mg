"""
Ovo je skripta u kojoj su testirane razne varijante MS-TCN modela sa i bez mogućnosti maskiranja.
Bilo je potrebno detljno testirati unaprijedni i unatražni korak kroz maskiranje.
Autori članka su napravili previd, jer rade maskiranje tek kod funkcije gubitka,
a potrebno je raditi maskiranje svakog sloja ako želimo dobiti jednak izlaz modela
za isto opažanje neovisno od nadopune. Tj. u njihovoj implementacij prolaz unaprijed
bi za isto opažanje ovisno o tome da li je nadopunjeno dao različite vrijednosti
što nije korektno!!!
"""

#%%Biblioteke
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import tensorflow as tf

#%%Klasa koja će omogućiti dodavanje maske izlazu iz Conv1D sloja
#Ovo je stara verzija koja u sebi nije sadržavala dodatan masking sloj
class MaskConv1D(layers.Layer):
  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Omogućava propagiranje maske
        self.supports_masking = True
    
    #Omogućava prijem maske, kroz argument "mask"
    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        masked_inputs = inputs * broadcast_float_mask   
        return masked_inputs
    
    #Omogućava spremanje i podizanja sloja sa pohranjenim argumentima konstruktora
    def get_config(self):
        base_config = super().get_config()
        return base_config

#%%Subclassing implementacija pomoćnih modula MS-TCN++ modela - bez maskiranja
#Dilatirani rezidualni modul
class DilatedResidualModule(layers.Layer):
   
    #Definicija elemenata sloja/modula
    def __init__(self, filters, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dilation_rate = dilation_rate
        
        #Ova komponenta kroz stopu dilatacije omogućava veće receptivno polje
        #uz isti broj parametara, bitno zbog hvatanja vremenskog konteksta u opažanjima
        self.conv_dilated = layers.Conv1D(filters, kernel_size=3, padding="same",
                                          dilation_rate=dilation_rate,
                                          activation="relu")
        
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        
        self.dropout = layers.Dropout(0.5)
    
    #Prolaz unaprijed kroz sloj
    def call(self, inputs):
        x = self.conv_dilated(inputs)
        x = self.conv_1x1(x)
        x = self.dropout(x)
        return inputs + x
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitno kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, 
                "dilation_rate": self.dilation_rate}

#Modul za generiranje inicijalnih predikcija - bez maskiranja
class PredictionGeneration(layers.Layer):
  
    def __init__(self, num_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        
        #Svrha ove komponente je podešavanje dimenzionalnosti ulaza
        self.conv_in = layers.Conv1D(filters, kernel_size=1)
        
        #Dualni dilatirani sloj
        self.conv_dilated_1 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**(num_layers - 1 - i)) for i in range(num_layers)]
        
        self.conv_dilated_2 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**i) for i in range(num_layers)]
        
        #Spajanje dualnih komponenti
        self.conv_fusion = [layers.Conv1D(filters, kernel_size=1, activation="relu") for i in range(num_layers)]
        
        self.dropout = layers.Dropout(0.5)
        
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1)
    
    def call(self, inputs):
        x = self.conv_in(inputs)
        
        for i in range(self.num_layers):
            shortcut = x
            dilated_1 = self.conv_dilated_1[i](x)
            dilated_2 = self.conv_dilated_2[i](x)
            concat = layers.concatenate([dilated_1, dilated_2], axis=-1)
            fusion = self.conv_fusion[i](concat)
            x = self.dropout(fusion)
            x = layers.add([x, shortcut])
        
        output = self.conv_out(x)
        
        return output
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes}

#Modul za rafiniranje predikcija - bez maskiranja
class Refeinment(layers.Layer):

    def __init__(self, num_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        
        #Podešavanje dimenzionalnosti ulaza
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        #U originalnoj implementaciji oni koriste dijeljene težine tj. isti modul više
        #puta, time smanjuju broj parametara, ali gube na točnosti modela
        self.dilated_residual_blocks = [DilatedResidualModule(filters, dilation_rate=2**i) 
                                        for i in range(num_layers)]
        
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1)
        
    def call(self, inputs):
        x = self.conv_1x1(inputs)
        
        for block in self.dilated_residual_blocks:
            x = block(x)
        
        output = self.conv_out(x)
        
        return output
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes}
    
#Funkcija za izgradnju više etapnog temporalnog konvolucijskog modela - bez maskiranja
def build_ms_tcn_model_no_mask(input_shape=(None,2048), num_layers_PG=11, num_layers_R=10, 
                    R_stages=3, shared_R=False, filters=64, num_classes=10):
 
    K.clear_session()
    #Nakon testiranja ovo makni!!!
    tf.random.set_seed(42)
    
    #Definicija elementa
    inputs = layers.Input(shape=input_shape)
    PG = PredictionGeneration(num_layers_PG, filters, num_classes)
    
    if shared_R:
        shared_refeinment = Refeinment(num_layers_R, filters, num_classes)
        Rs = [shared_refeinment for _ in range(R_stages)]
    else:
        Rs = [Refeinment(num_layers_R, filters, num_classes) for _ in range(R_stages)]
    
    #Generiranje modela
    x = PG(inputs)
    x = layers.Activation("softmax")(x)
    
    #Model ima izlaz za svaku etapu
    outputs = [x]
    for R in Rs:
        x = R(x)
        x = layers.Activation("softmax")(x)
        outputs.append(x)
          
    return Model(inputs=inputs, outputs=outputs)

#%%Maskiranje svega
class MaskConv1D_v2(layers.Layer):
  
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

#Dilatirani rezidualni modul
class MaskDilatedResidualModule(layers.Layer):
   
    #Definicija elemenata sloja/modula
    def __init__(self, filters, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dilation_rate = dilation_rate
        
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D_v2()
        #Ova komponenta kroz stopu dilatacije omogućava veće receptivno polje
        #uz isti broj parametara, bitno zbog hvatanja vremenskog konteksta u opažanjima
        self.conv_dilated = layers.Conv1D(filters, kernel_size=3, padding="same",
                                          dilation_rate=dilation_rate,
                                          activation="relu")
        
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        
        self.dropout = layers.Dropout(0.5)
    
    #Prolaz unaprijed kroz sloj
    def call(self, inputs):
        mask = self.masking.compute_mask(inputs)
        masked_inputs = self.masking(inputs)
        
        x = self.conv_dilated(masked_inputs)
        x = self.conv_mask(x, mask=mask)
        #x = self.masking(x)
        
        x = self.conv_1x1(x)
        x = self.conv_mask(x, mask=mask)
        #x = self.masking(x)
        
        x = self.dropout(x)
        output = layers.add([masked_inputs, x])
        #output = self.conv_mask(x, mask=mask)
        #output = self.masking(x)
        
        return output
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitno kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters": self.filters, 
                "dilation_rate": self.dilation_rate}


#%%Modul za generiranje inicijalnih predikcija
class MaskPredictionGeneration(layers.Layer):
  
    def __init__(self, num_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D_v2()
        
        #Svrha ove komponente je podešavanje dimenzionalnosti ulaza
        self.conv_in = layers.Conv1D(filters, kernel_size=1)
        
        #Dualni dilatirani sloj
        self.conv_dilated_1 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**(num_layers - 1 - i)) for i in range(num_layers)]
        
        self.conv_dilated_2 = [layers.Conv1D(filters, kernel_size=3, padding="same",
                                             dilation_rate=2**i) for i in range(num_layers)]
        
        #Spajanje dualnih komponenti
        self.conv_fusion = [layers.Conv1D(filters, kernel_size=1, activation="relu") for i in range(num_layers)]
        
        self.dropout = layers.Dropout(0.5)
        
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")
    
    def call(self, inputs):
        mask = self.masking.compute_mask(inputs)
        
        x = self.conv_in(inputs)
        x = self.masking(x)
        
        for i in range(self.num_layers):
            shortcut = x
            
            dilated_1 = self.conv_dilated_1[i](x)
            dilated_1 = self.conv_mask(dilated_1, mask=mask)
            #dilated_1 = self.masking(dilated_1)
            
            dilated_2 = self.conv_dilated_2[i](x)
            dilated_2 = self.conv_mask(dilated_2, mask=mask)
            #dilated_2 = self.masking(dilated_2)
            
            concat = layers.concatenate([dilated_1, dilated_2], axis=-1)
            
            fusion = self.conv_fusion[i](concat)
            fusion = self.conv_mask(fusion, mask=mask)
            #fusion = self.masking(fusion)
            
            x = self.dropout(fusion)
            x = layers.add([x, shortcut])
        
        output = self.conv_out(x)
        output = self.conv_mask(output, mask=mask)
        #output = self.masking(output)
        
        return output
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes}

#%%Modul za rafiniranje predikcija
class MaskRefeinment(layers.Layer):

    def __init__(self, num_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.filters = filters
        self.num_classes = num_classes
        
        self.masking = layers.Masking(mask_value=0.)
        self.conv_mask = MaskConv1D_v2()
        #Podešavanje dimenzionalnosti ulaza
        self.conv_1x1 = layers.Conv1D(filters, kernel_size=1)
        #U originalnoj implementaciji oni koriste dijeljene težine tj. isti modul više
        #puta, time smanjuju broj parametara, ali gube na točnosti modela
        self.dilated_residual_blocks = [MaskDilatedResidualModule(filters, dilation_rate=2**i) 
                                        for i in range(num_layers)]
        #Softmax je dodan naknadno
        self.conv_out = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")
        
    def call(self, inputs):
        mask = self.masking.compute_mask(inputs)
        
        x = self.conv_1x1(inputs)
        x = self.masking(x)
        
        for block in self.dilated_residual_blocks:
            x = block(x)
        
        output = self.conv_out(x)
        output = self.conv_mask(output, mask=mask)
        #output = self.masking(output)
        
        return output
        
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_layers": self.num_layers, 
                "filters": self.filters, "num_classes": self.num_classes}

#%%
def build_ms_tcn_model_simple_mask(input_shape=(None,2048), mask_value=0., num_layers_PG=11, num_layers_R=10, 
                    R_stages=3, shared_R=False, filters=64, num_classes=10):
 
    K.clear_session()
    tf.random.set_seed(42)
    
    #Definicija elementa
    masking = layers.Masking(mask_value=mask_value)
    #mask_conv = MaskConv1D_v2(mask_value=0.)
    inputs = layers.Input(shape=input_shape)
    PG = MaskPredictionGeneration(num_layers_PG, filters, num_classes)
    
    if shared_R:
        shared_refeinment = MaskRefeinment(num_layers_R, filters, num_classes)
        Rs = [shared_refeinment for _ in range(R_stages)]
    else:
        Rs = [MaskRefeinment(num_layers_R, filters, num_classes) for _ in range(R_stages)]
    
    #Generiranje modela
    
    #mask = masking.compute_mask(inputs)
    
    x = masking(inputs)
    
    x = PG(x) 
    #x = layers.Activation("softmax")(x)
    #x = mask_conv(x, mask=mask)
    #x = masking(x)
    
    #Model ima izlaz za svaku etapu, ali ću ga testirati samo sa jednim izlazom zbog lakšeg praćenja rezultata
    outputs = [x]
    for R in Rs:
        x = R(x)
        #x = layers.Activation("softmax")(x)
        #x = mask_conv(x, mask=mask)
        #x = masking(x)
    
        outputs.append(x)
    #outputs = mask_conv(x, mask=mask)
    #outputs = masking(outputs)
   
    return Model(inputs=inputs, outputs=outputs)


#%%Funkcionalna implementacija

#Dilatirani rezidualni modul
def _dilated_residual_module(inputs, filters, dilation_rate):
    
    #Preskočna veza
    shortcut = inputs
    
    #Glavna grana
    conv_dilated = layers.Conv1D(filters, kernel_size=3, padding="same", 
                                 dilation_rate=dilation_rate, activation="relu")(inputs)
    
    conv_1x1 = layers.Conv1D(filters, kernel_size=1)(conv_dilated)
    
    dropout = layers.Dropout(0.5)(conv_1x1)
    
    #Zbrajanje preskočne i glavne grane 
    output = layers.add([dropout, shortcut])
    
    return output

#Modul za generiranje inicijalnih predikcija
def _prediction_generation_module(inputs, num_layers, filters, num_classes):
       
    x = layers.Conv1D(filters, kernel_size=1)(inputs)
    
    for i in range(num_layers):
        
        shortcut = x
        
        conv_dilated_1 = layers.Conv1D(filters, kernel_size=3, padding="same",
                                       dilation_rate=2**(num_layers - 1 - i))(x)
        
        
        conv_dilated_2 = layers.Conv1D(filters, kernel_size=3, padding="same",
                                       dilation_rate=2**i)(x)
        
        #Oprezno: PyTorch cat i tf.concatenate drugačije definiraju os po kojoj naslagujemo!!! u Torch-u je axis=1, jer još
        #nije definirana batch dimenzija, a kod TF je axis=-1, jer ovdje već postoji batch dimenzija tj.
        #Torch ([time, filters]), a TF je [batch, time, filters], CILJ je povećati dimenzionalnost značajki,
        #a ne broj vremenskih koraka, jer bi "add" podignuo iznimku zbog nekompatibilnih dimenzija
        concat = layers.concatenate([conv_dilated_1, conv_dilated_2], axis=-1)
        
        conv_fusion = layers.Conv1D(filters, kernel_size=1)(concat)
        
        x = layers.Activation("relu")(conv_fusion)
        
        x = layers.Dropout(0.5)(x)
        
        x = layers.add([x, shortcut])
        
    output = layers.Conv1D(num_classes, kernel_size=1)(x)
    
    return output

#Modul za rafiniranje predikcija
def _refeinment_module(inputs, num_layers, filters, num_classes):
    
    x = layers.Conv1D(filters, kernel_size=1)(inputs)
    
    for i in range(num_layers):
        x = _dilated_residual_module(x, filters, dilation_rate=2**i)
    
    outputs = layers.Conv1D(num_classes, kernel_size=1)(x) 
    
    return outputs

#Funkcija za izgradnju više etapnog temporalnog konvolucijskog modela

def _build_ms_tcn_model(input_shape=(None, 2048), mask_value=0., num_layers_PG=11, num_layers_R=10, 
                            R_stages=3, filters=64, num_classes=10):
    
    K.clear_session()
    
    #Definicija ulaza
    inputs = layers.Input(shape=input_shape)
     
    #Glavna grana modela
    x = _prediction_generation_module(inputs, num_layers_PG, filters, num_classes)
    outputs = [x]
      
    #Petlja po broju etapa rafiniranja predikcije
    for stage in range(R_stages):
         
        x = layers.Activation("softmax")(x)
        
        x = _refeinment_module(x, num_layers_R, filters, num_classes)
        
        outputs.append(x)
    
    return Model(inputs=inputs, outputs=outputs)
#%%CCE + MSE truncated loss = Segmentation Loss

class SegmentationLoss(tf.keras.losses.Loss):
    '''Gubitak za problem video segmentacije. Sastoji se od kombinacije
    dva gubitka: klasične unakrsne entropije za problem klasifikacije te 
    srednjeg kvadrata odstupanja između log vjerojatnosti dva uzastopna vremenska
    koraka koji je ograničen (eng. "truncated") sa gornjom granicom (tau). Zadatak
    druge komponente gubitka je smanjiti prekomjernu segmentaciju tj. izgladiti gubitak
    .'''
    
    #Definiranje hiperparametara gubitka
    def __init__(self, lmbda=0.15, tau=4., num_classes=10, is_one_hot=False, **kwargs):
        self.lmbda = lmbda
        self.tau = tau
        self.num_classes = num_classes
        self.is_one_hot = is_one_hot
        super().__init__(**kwargs)
    
    #Izračun gubitka
    def call(self, y_true, y_pred):
        
        masked_steps = tf.cast(tf.reduce_all(tf.not_equal(y_pred, 0.), axis=-1), tf.float32)
        masked_steps = tf.expand_dims(masked_steps, axis=-1)
        
        #Osigurava numeričku stablinost, konkretno izbjegavamo log(0.) => nan
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-7, clip_value_max=1. - 1e-7)
        
        #Da li su podatci One Hot kodirani, ako ne kodiraj ih
        if not self.is_one_hot:
            y_true=(tf.one_hot(y_true, depth=self.num_classes))
        
        #Provjeri da li treba izračunati i mean ili to radi TF po batchevima
        s1 = tf.cast(tf.shape(y_true)[0], tf.float32)
        s2 =  tf.cast(tf.shape(y_true)[1], tf.float32)
        denominator = tf.multiply(s1, s2) 
        CCE_loss = - tf.divide(tf.math.reduce_sum(y_true * tf.math.log(y_pred)), denominator)
        
        delta_tc = tf.abs(tf.subtract(tf.math.log_softmax(y_pred[:, 1:,:]), tf.math.log.log_softmax(tf.stop_gradient(y_pred[:, :-1, :]))))
        truncated = tf.greater(delta_tc, self.tau)
        #Obavezno su true i false ishodi ista vrsta podatka npr tau i delta_tc su oboje float
        delta_tc_truncated = tf.where(truncated, self.tau,  delta_tc)
        T_MSE_loss = tf.math.reduce_mean(tf.square(delta_tc_truncated) * masked_steps[:, 1:])
        
        total_loss = tf.add(CCE_loss, tf.multiply(self.lmbda, T_MSE_loss))
        
        return total_loss
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitno kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "lmbda": self.lmbda, "tau": self.tau,
                "num_classes": self.num_classes, "is_one_hot": self.is_one_hot}

#Problem kod prve implementacije gubitka je da kod nadopunjenih batcheva ne zna kako prihvatiti masku

from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError

class SegmentationLoss_v2(tf.keras.losses.Loss):
    '''Gubitak za problem video segmentacije. Sastoji se od kombinacije
    dva gubitka: klasične unakrsne entropije za problem klasifikacije te 
    srednjeg kvadrata odstupanja između log vjerojatnosti dva uzastopna vremenska
    koraka koji je ograničen (eng. "truncated") sa gornjom granicom (tau). Zadatak
    druge komponente gubitka je smanjiti prekomjernu segmentaciju tj. izgladiti gubitak
    .'''
    
    #Definiranje hiperparametara gubitka
    def __init__(self, lmbda=0.15, tau=4., num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.lmbda = lmbda
        self.tau = tau
        self.num_classes = num_classes
        #Obavezno bez redukcije(SUM ili slično) nju radi automatski framework
        self.MSE_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.CCE_loss = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
          
    #Izračun gubitka
    def call(self, y_true, y_pred):
               
        #Pretvorba stvarnih oznaka u one-hot kodiranje
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        
        ##CCE gubitak##
        CCE_loss = self.CCE_loss(y_true, y_pred)
        
        ##T-MSE gubitak##
        y_pred_t1 = tf.math.log_softmax(tf.concat([y_pred[:, 1: ], tf.stop_gradient(y_pred[:, -1:])], axis=1), axis=-1) 
        y_pred_t0 = tf.math.log_softmax(tf.concat([y_pred[:, :-1], y_pred[:, -1:]], axis=1), axis=-1)
        T_MSE_loss = tf.clip_by_value(self.MSE_loss(y_pred_t1, tf.stop_gradient(y_pred_t0)), 0., tf.square(self.tau))
        
        return  tf.add(CCE_loss, tf.multiply(self.lmbda, T_MSE_loss))
    
    #Metoda koja omogućava pohranu argumenata konstruktora, 
    #bitno kod podizanja/spremanja modela
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "lmbda": self.lmbda, "tau": self.tau,
                "num_classes": self.num_classes}
    
