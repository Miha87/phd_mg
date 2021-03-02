#%%Biblioteke
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np

#%%Funkcija za izgradnju optimiziranog pipeline-a, za treniranje modela
#na slikama ili video zapisima

def build_train_pipeline(filenames, parser_fun, batch_size, padded_batch=False, 
                   padded_shapes=([None, None], [None]), padding_values=(0.,0), 
                   drop_remainder=False, add_options=False, compression=None):
    
    '''Funkcija za generiranje optimiziranog pipeline-a za podatke u TFRecord formatu.
    
    Argumenti
    ---------
    filenames: str
        Moguće je zadati punu putanju do datoteka, ili 
        glob obrazac za matchiranje/prepoznavanje datoteka u određenim direktorijima, 
        npr. "*.tfrecord".
    
    parser_fun: fun
        Funkcija za parsiranje TFrecord datoteke u format pogodan za daljnju obradu
        primjenom modela.

    batch_size: int
       Veličina mini grupe za učenje modela.

    padded_batch: bool
        Da li je potrebno nadopuniti broj vremenskih koraku za opažanja u mini-grupi. 
        Ovo ima smisla koristiti npr. kod video zapisa različitih duljina.
        Zadana vrijednost je False.

    padded_shapes: None/int/tuple
        Ako je nadopuna specifične dimenzije postavljena na None tada će u toj dimenziji
        sva opažanja biti nadopunjena na duljinu najduže komponente.
        Inače je nužno pratiti strukturu ulaza, npr. tuple sa dvije liste kao argument. 
        Prva lista sadrži informaciju o dimenzijama nadopune opažanja, a druga o 
        dimenzijama nadopune oznaka. Zadana vrijednost je ([None, None], [None]).

    padded_values: None/float/int/tuple
        Vrijednost nadopune. 
        Ako je None, tada je vrijednost nadopune u svim dimenzijama opažanja i oznaka 0 
        u slučaju da se radi o numeričkim vrijednostima ili "" u slučaju stringa. 
        Ako zadamo skalarnu vrijednost ona se koristi po svim dimenzijama opažanja i oznake. 
        Ako zadamo tuple skalara, onda se prvi član tuple-a koristi za sve dimenzije opažanja,
        a drugi član za oznake, pri čemu nadopuna mora biti isti tip podatka kao i opažanja i oznake
        npr. (float32, int16). 
        U svim ostalim slučajevima struktura nadopune mora pratiti  strukturu ulaznog skupa podataka i tip podataka. 
        Oznaka nadopuna za y mora biti 0, (inače ne radi: Vraća NaN kod 
        izračuna funkcije gubitka za SparseCategoricalCrossentropy)  bez obzira da li 
        u skupu podataka postoji oznaka 0 (nema utjecaj jer će nadopune biti maskirane!).

    drop_remainder: bool
        Odbaci mini-grupu koja nije iste veličine kao ostale, npr. ako je zadnja mini grupa
        u epohi manje veličine ne koristimo je. Zadana vrijednost je False.
    
    dodaj_opcije: bool
        Dodavanje statičkih optimizacija pipeline-a, zadana vrijednost je False.
    
    compression: str
        Da li su podatci komprimirani, npr. u GZIP formatu, zadana vrijednost je None.
    
    Povratna vrijednost
    -------------------
    dataset: tf.data.Dataset
        Instanca tf.data.Dataset klase pogodna za obradu modelom

    '''
    #Vraća shufflirane putanje do TFRecorda
    dataset = tf.data.Dataset.list_files(filenames)
    
    #Parlelno čita više Exampleove iz više TFRecorda
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression),
                                 cycle_length=tf.data.experimental.AUTOTUNE,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #Dodatno shuffliranje opažanja
    dataset = dataset.shuffle(buffer_size=1000)
   
    #Koju vrstu mini grupa koristimo
    if padded_batch:
        #Kod padded_batch-a, map ide prije, jer parser funkcija radi na opažnjima različitih duljina
        #pa to ne vektoriziramo 
        dataset = dataset.map(parser_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, 
                                       padding_values=padding_values, drop_remainder=drop_remainder)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        #Map poslije batcha, zato je potrebno imati vektorizirani parser jer mora raditi na batchu
        dataset = dataset.map(parser_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #Vraća mini grupu i priprema slijedeću (na CPU), kako GPU ne bi bio bez posla 
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    #Statičke optimizacije
    if add_options:
        options = tf.data.Options()
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_parallelization = True
        dataset = dataset.with_options(options)
        
    return dataset

#%%Funkcija za izgradnju jednostavnog pipeline-a za evaluaciju modela
def build_test_pipeline(filenames, parser_fun, batch_size, padded_batch=False,
                        padded_shapes=([None, None], [None]), 
                        padding_values=(0.,0)):
    '''Generira dataset za predikciju svakog opažanja. U slučaju da je 
    ulaz vremenska serija(npr. video), argument padded_batch mora biti True!.
    
    Argumenti
    ---------
    filenames: str
        Moguće je zadati punu putanju do datoteka, ili 
        glob obrazac za matchiranje/prepoznavanje datoteka u određenim direktorijima, 
        npr. "*.tfrecord".
    
    parser_fun: fun
        Funkcija za parsiranje TFrecord datoteke u format pogodan za daljnju obradu
        primjenom modela.
   
    padded_batch: bool
        Da li je potrebno nadopuniti broj vremenskih koraku za opažanja u mini-grupi. 
        Ovo ima smisla koristiti npr. kod video zapisa različitih duljina.
        Zadana vrijednost je False.

    padded_shapes: None/int/tuple
        Ako je nadopuna specifične dimenzije postavljena na None tada će u toj dimenziji
        sva opažanja biti nadopunjena na duljinu najduže komponente.
        Inače je nužno pratiti strukturu ulaza, npr. tuple sa dvije liste kao argument. 
        Prva lista sadrži informaciju o dimenzijama nadopune opažanja, a druga o 
        dimenzijama nadopune oznaka. Zadana vrijednost je ([None, None], [None]).

    padded_values: None/float/int/tuple
        Vrijednost nadopune. 
        Ako je None, tada je vrijednost nadopune u svim dimenzijama opažanja i oznaka 0 
        u slučaju da se radi o numeričkim vrijednostima ili "" u slučaju stringa. 
        Ako zadamo skalarnu vrijednost ona se koristi po svim dimenzijama opažanja i oznake. 
        Ako zadamo tuple skalara, onda se prvi član tuple-a koristi za sve dimenzije opažanja,
        a drugi član za oznake, pri čemu nadopuna mora biti isti tip podatka kao i opažanja i oznake
        npr. (float32, int16). 
        U svim ostalim slučajevima struktura nadopune mora pratiti  strukturu ulaznog skupa podataka i tip podataka. 
        Oznaka nadopuna za y mora biti 0, (inače ne radi: Vraća NaN kod 
        izračuna funkcije gubitka za SparseCategoricalCrossentropy)  bez obzira da li 
        u skupu podataka postoji oznaka 0 (nema utjecaj jer će nadopune biti maskirane!)
        
    Povratna vrijednost
    -------------------
    dataset: tf.data.Dataset
        Instanca tf.data.Dataset klase pogodna za generiranje predikcija iz modela.
    '''
    dataset = tf.data.TFRecordDataset(filenames)
    
    if padded_batch:
        
        dataset = dataset.map(parser_fun)
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes,
                                       padding_values=padding_values)
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parser_fun)
        
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset    

#%%Funkcija za testiranje brzine pipeline-a

def pipeline_timer(dataset, num_iterations=20):
    '''Mjeri izvođenja definiranog broja iteracije iz 
    zadanog dataset-a. Prima tf.data.Dataset te vraća vrijeme
    potrebno da vrati mini grupu.'''
    
    print("Početak mjerenja")
    start_time = timer()
    process_times = []
    for example in dataset:
        end_time = timer()
        process_times.append(end_time - start_time)
        start_time = end_time
        num_iterations -= 1
        if num_iterations == 0:
            break
    print("Kraj mjerenja")
    print("Prosječno vrijeme po batchu:", f"{np.mean(process_times)} sekundi")
    return process_times

