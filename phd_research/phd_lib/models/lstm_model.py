#%%Biblioteke
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf

#%%Definiranje funkcije za generiranje jednosmjernih i dvosmjernih Lstm model-a

def build_lstm_model(input_shape=(None, 2048), mask_value=0., bidirect=False, num_lstm_layers=1, 
                     lstm_units=512, lstm_dropout=0.5, lstm_recurrent_dropout=0., 
                     num_fc_layers=1, fc_reg=0., fc_units=256, fc_dropout=0.5, 
                     num_classes=10, seed=42):
    """
    Definiranje jednosmjernih i dvosmjernih LSTM modela, 
    sa dodatkom jednog ili dva potpuno povezana sloja.

    Argumenti:
    ----------
    input_shape: tuple
        Dimenzionalnost ulaznog vremenskog niza. U slučaju da se koriste nizovi
        različitih duljina prva dimenzija je None. Preddefinirana vrijednost je (None, 2048).
        
    mask_value: int/float
        Vrijednost nadopune vremenskog niza koju je potrebno zanemariti kod
        kalkulacija. Mora odgovarati tipu podataka u vremenskom nizu. 
        Preddefinirana vrijednost je 0..
        
    bidirect: bool
        Da li koristimo dvosmjerni LSTM. Preddefinirana vrijednost je False.
        
    num_lstm_layers: int
        Broj LSTM slojeva. Preddefinirana vrijednost je 1.
        
    lstm_units: int
        Broj neurona u skrivenom sloju LSTM-a. Preddefinirana vrijednost je 512.
        
    lstm_dropout: float
        Stopa ugašenih neurona za ulazne matrice LSTM-a. Mora biti u
        rasponu [0, 1]. Preddefinirana vrijednost je 0.5.
        
    lstm_recurrent_dropout: float
        Stopa ugašenih neurona za matrice skrivenog stanja LSTM-a. 
        Mora biti u rasponu [0, 1]. U slučaju da je vrijednost ovog argumenta 
        različita od 0, nije moguće koristit CuDNN implementaciju LSTM sloja te 
        će proces učenja biti sporiji. Preddefinirana vrijednost je 0..
        
    num_fc_layers: int
        Broj potpuno povezanih slojeva. Može biti 1 ili 2. 
        Preddefinirana vrijednost je 1.
        
    fc_reg: float
        Snaga L2 regularizacije potpuno povezanih slojeva. 
        Preddefinirana vrijednost je 0.
        
    fc_units: int
        Broj neurona u dodatnom potpuno povezanom sloju. Ako je broj potpuno
        povezanih slojeva jednak 1, onda je ovaj argument zanemaren! 
        Preddefinirana vrijednost je 256.
        
    fc_dropout: int
        Stopa ugašenih neurona za dodatni potpuno povezani sloju. Ako je broj potpuno
        povezanih slojeva jednak 1, onda je ovaj argument zanemaren! 
        Preddefinirana vrijednost je 0.5.
        
    num_classes: int
        Broj klasa za koji učimo model, odgovara broju neurona u zadnjem potpuno povezanom
        sloju. Preddefinirana vrijednost je 10.
        
    seed: int
        Za reprodukciju rezultata.

    Povratna vrijednost:
    --------------------
    model : Model
        Vraća instancu klase Model.

    """ 
    K.clear_session()
    
    tf.random.set_seed(seed)
    
    #Provjera ulaza
    if lstm_dropout < 0.0 or lstm_dropout > 1.0:
        raise ValueError("Dropout mora biti u rasponu [0, 1]!")
    if lstm_recurrent_dropout < 0.0 or lstm_recurrent_dropout > 1.0:
        raise ValueError("Dropout mora biti u rasponu [0, 1]!")
    if fc_dropout < 0.0 or fc_dropout > 1.0:
        raise ValueError("Dropout mora biti u rasponu [0, 1]!")
    if num_fc_layers not in [1, 2]:
        raise ValueError("Broj potpunog povezanih slojeva mora biti 1 ili 2 !!!")
    
    #Ulaz i maskiranje nadopuna
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=mask_value)(inputs)
    
    #Definiranje komponenata LSTM slojeva
    LSTM_layers = [layers.LSTM(lstm_units, dropout=lstm_dropout, 
                               recurrent_dropout=lstm_recurrent_dropout,
                               return_sequences=True) for i in range(num_lstm_layers)]
    
    #Petlja po LSTM slojevima
    for lstm_layer in LSTM_layers:
        #Da li koristimo dvosmjerni ili jednosmjerni LSTM
        if bidirect:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
             
    #Da li imamo dva potpuno povezana sloja
    if num_fc_layers == 2:
        x = layers.Dense(fc_units, activation="relu", 
                         kernel_regularizer=regularizers.l2(fc_reg))(x)
        x = layers.Dropout(fc_dropout)(x)
    
    #Završni potpuno povezani sloj
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(fc_reg))(x)
    
    model = Model(inputs=inputs, outputs=outputs) 
    
    return model



