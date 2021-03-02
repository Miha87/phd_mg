'''REFERENCA
Implementacija prema članku:
    Kaiming He et al. “Identity Mappings in Deep Residual Networks”. In: CoRR abs/1603.05027
(2016)
Dodatan izvor:
    Adrian Rosebrock, "Pyimagesearch" In: https://www.pyimagesearch.com 

TODO: U budućnosti rezidualni modul pretvoriti u instancu tf.keras.layers.Layer klase
kako bi imao uniformno sučelje za funkcionalni API.
'''

#%%Biblioteke
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

#%%Privatna funkcija sa definicijom rezidualnog modula

def _residual_module(inputs, filters, stride, channel_dim, reduce_dim=False,
                     reg=0.0001, bn_eps=2e-5, bn_momentum=0.9):
    """
    Definicija rezidualnog modula, pri čemu se modul sastoji 
    od bottleneck grane i preskočne grane. Ova funkcija je privatna jer 
    samostalno ne radi ako je reduce_dim=False za prvi blok, zbog
    nekompatibilnih dimenzija u layers.add sloju.
    
    Argumenti:
    ----------
    inputs: tensor
        Ulaz u rezidualni blok.
        
    filters: int
        Broj filtera koji će učiti zadnji konvolucijski sloj u bottelneck-u.
    
    stride: int
        Posmak u operaciji konvolucije.
    
    channel_dim: int
        Koja os predstavlja oznaku kanala (obično je prva ili zadnja). Bitno
        kod primjene BatchNorm-a.
    
    reduce_dim: bool
        Da li se u rezidualno bloku treba napraviti smanjenje prostorne dimenzionalnosti. 
        Preddefinirana vrijednost je False.
    
    reg: float
        Stupanje regularizacije za sve konvolucijske slojeve. Preddefinirana
        vrijednost je 0.0001.
    
    bn_eps: float
        Prilikom normalizacije u BN slojevima konstanta koja osigurava da 
        ne dođe do dijeljenja sa nulom. Preddefinirana vrijednost je 2e-5.
    
    bn_momentum: float
        Momentum pomičnog prosjeka kod BN slojeva. Preddefinirana 
        vrijednost je 0.9.

    Povratna vrijednost:
    --------------------
    output: tensor
    Transformirani ulazni podatci.
    
    """
    #Inicijalizacija preskočne veze
    shortcut = inputs
    
    #Prvi sloj sa 1x1 filterima i pred-aktivacijom(bn + relu)
    bn1 = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps,
                                    momentum=bn_momentum)(inputs)
    act1 = layers.Activation("relu")(bn1)
    #Prvi konv. sloji uči 4x manje filtera u odnosu na zadnji, ne treba nam
    #vektor pristranosti jer koristimo BN sloj
    conv1 = layers.Conv2D(int(filters * 0.25), (1, 1), use_bias=False,
                          kernel_regularizer=regularizers.l2(reg))(act1)
    
    #Drugi sloj sa 3x3 filterima
    bn2 = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps,
                                    momentum=bn_momentum)(conv1)
    act2 = layers.Activation("relu")(bn2)
    conv2 = layers.Conv2D(int(filters * 0.25), (3, 3), strides=stride, padding="same",
                          use_bias=False, kernel_regularizer=regularizers.l2(reg))(act2)
    
    #Treći sloj sa 1x1 filterima
    bn3 = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps,
                                    momentum=bn_momentum)(conv2)
    act3 = layers.Activation("relu")(bn3)
    conv3 = layers.Conv2D(filters, (1, 1), use_bias=False, 
                          kernel_regularizer=regularizers.l2(reg))(act3)
    
    #Da li smanjujemo prostornu dim. ulaza
    if reduce_dim:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, 
                                        kernel_regularizer=regularizers.l2(reg))(act1)
    
    #Zbrajanje preskočne veze i transformacije bottleneck grane
    output = layers.add([conv3, shortcut])
    
    return output

#%%Funkcija za izgradnju dubokog modela sa rezidualnim blokovima
          
def build_resnet_model(h, w, depth, stages, filters, num_classes, reduce_in_dim=False, reg=0.0001, bn_eps=2e-5,
          bn_momentum=0.9, seed=42):
    """
    Funkcija za izgradnju modela sa rezidualnim blokovima.
    
    PRIMJER
    '''python
        model = build_resnet_model(224, 224, 3, stages=(3, 4, 6), filters=(64, 128, 256, 512), num_classes=10)
    
    Ulazna dimenzija je (224, 224, 3) pri čemu model uči 10 klasa,
    Prvi konvolucijski sloj uči 64 filtera, slijedeća 3 rezidualna bloka uče po 128 filtera,
    slijedeća 4 rezidualna bloka uče po 256 filtera te zadnjih 6 rezidualnih blokova uče po
    512 filtera.
    '''

    Argumenti:
    ----------
    h: int
        Visina ulazne slike.
   
    w: int
        Širina ulazne slike.
    
    depth: int
        Broj kanala ulazne slike.
    
    stages: tuple(int)
        Broj etapa tj. rezidualnih blokova koji uči jednaka broj filtera.
        
    filters: tuple(int)
        Broj filtera koji uči prvi konvolucijski sloj te rezidualni blokovi.
    
    num_classes: int
        Broj različitih klasa koje je potrebno naučiti, izlazna dimenzija.
    
    reduce_in_dim: bool
        Da li se prije ulaza u rezidualne blokove treba napraviti smanjenje 
        prostorne dimenzionalnosti. 
        Preddefinirana vrijednost je False.
        
    reg: float
        Stupanje regularizacije za sve konvolucijske slojeve. Preddefinirana
        vrijednost je 0.0001.
        
    bn_eps: float
        Prilikom normalizacije u BN slojevima konstanta koja osigurava da 
        ne dođe do dijeljenja sa nulom. Preddefinirana vrijednost je 2e-5.
    
    bn_momentum: float
        Momentum pomičnog prosjeka kod BN slojeva. Preddefinirana 
        vrijednost je 0.9.
        
    seed: int
        Za reprodukciju rezultata.

    Povratna vrijednost:
    --------------------
    model: Model
    Vraća instancu klase Model.
    
    """
    K.clear_session()
    
    tf.random.set_seed(42)
    
    #Provjera ulaza, kako bi mogli raditi i sa plitkim modelima (1 etapa)
    if isinstance(stages, int):
        stages=[stages]
    
    #inicijalizacija ulaznih dimenzija i definiranje zadnje osi kao dim. kanala
    input_shape = (h, w, depth)
    channel_dim = -1
    
    #u slučaju da su kanali prva os kod dimenzije slike
    if K.image_data_format() == "channels_first":
        input_shape = (depth, h, w)
        channel_dim = 1
    
    #Definiranje ulaza modela
    inputs = layers.Input(shape=input_shape)
    #Normalizacija ulaz kroz BN sloj
    x = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps, 
                                  momentum=bn_momentum)(inputs)
   
    #Da li reduciramo prostornu dimenziju ulaza
    if not reduce_in_dim:
        x = layers.Conv2D(filters[0], (3, 3), use_bias=False,
			padding="same", kernel_regularizer=regularizers.l2(reg))(x)
    else:
        x = layers.ZeroPadding2D((3,3))(x)
        x = layers.Conv2D(filters[0], (7, 7), strides=(2, 2), use_bias=False,
                          padding="valid",
                          kernel_regularizer=regularizers.l2(reg))(x)
        x = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps, 
                                  momentum=bn_momentum)(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        
    #Petlja po etapama
    for i in range(len(stages)):
        
        #Za svaku etapu osim PRVE, u prvom rezidualnom bloku trenutne etape
        #napravi poduzorkovanje u bottleneck grani primjenom posmaka (2, 2),
        #zbog ovoga ne moramo koristiti slojeve sažimanja ("pooling")
        stride = (1, 1) if i == 0 else (2, 2)
    
        #Za svaku etapu, u prvom rezidualnom bloku napravi reduciranje dimenzionalnosti
        #(broja "kanala") preskočne grane, bez ovoga zbrajanje bottleneck i preskočne 
        #grane NIJE MOGUĆE!!
        x = _residual_module(x, filters[i + 1], stride, channel_dim,
                             reduce_dim=True, bn_eps=bn_eps, 
                             bn_momentum=bn_momentum)
        
        #Petlja po broju slojeva, -1 jer smo prvi rez. blok etape već napravili
        for j in range(stages[i] - 1):
            x = _residual_module(x, filters[i + 1], (1, 1), channel_dim,
                                 bn_eps=bn_eps, bn_momentum=bn_momentum)
    
    #Primjeni BN, RELU, AVGPOOL
    x = layers.BatchNormalization(axis=channel_dim, epsilon=bn_eps,
                                  momentum=bn_momentum)(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling2D((7, 7))(x)
    
    #Klasifikacija
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(reg))(x)
    outputs = layers.Activation("softmax")(x)
    
    #Kreiraj model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
  
        
        