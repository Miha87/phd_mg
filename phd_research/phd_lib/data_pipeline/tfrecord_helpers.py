#%%Biblioteke
import tensorflow as tf

#%%Helper funkcije

#Tri funkcije preuzete sa službene TF stranice za stvaranje byte, float i int lista
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#Funkcija za prebacivanje slike u ndarray formatu u byte string
def _image2bytes(image):
    '''Pretvara sliku iz ndarray formata u byte string'''
    #Ne enkodiramo u jpeg, datoteka će biti teža (uint8), ali će biti brže čitanje
    return image.tobytes()

#Funkcija za kreiranje FeatureList-a iz niza sličica
def _image_seq2FeatureList(image_seq):
    '''Ulaz je niz slika spremljenih u array-like strukturi pri čemu prva dimenzija
    odgovara broju slika, a izlaz je FeatureList pogodan za kreiranje SequenceExample-a.'''
    
    #Svaku sliku u videu pretvori u Feature
    image_feats = [_bytes_feature(_image2bytes(image)) for image in image_seq]
    
    #Listu Feature-a pretvori u FeatureList
    feat_list = tf.train.FeatureList(feature=image_feats)

    return feat_list

#Funkcija za kreiranje FeatureList-a iz niza oznaka
def _labels2FeatureList(labels):
    '''Ulaz je niza oznaka jednog videa u array-like strukturi, a izlaz je 
    FeatureList pogodan za kreiranje SequenceExample-a.'''
      
    #Svaku oznaku pretvori u Feature
    label_feats = [_int64_feature(label) for label in labels]
    
    #Listu Feature-a pretvori u FeatureList
    feat_list = tf.train.FeatureList(feature=label_feats)

    return feat_list

#Čitač oznaka sličica iz videa
def read_labels(label_path):
    '''Pročitaj oznake za zadanu putanju, oznake su .txt formatu 
    sa strukturom [video_id, oznake].'''

    with open(label_path + '.txt', "r") as f:
        #Uzmi samo oznake, izbaci video_id
        labels = f.readline().split(",")[1:]
        #Pretvori oznake u int listu
        labs = [int(label) for label in labels]
        return labs

#%% Funkcije za kreiranje serijaliziranih Example-ova iz slika i videa
def image2example(image_id, image, label):
    '''Radi pretvorbu id-a slike, slike i oznake u serijalizirani Example.'''
    
    #Kreiramo Feature dict
    feature_dict = {
        #image_id ima strukturu kadar/podjela_podataka/video_id/slika-####.npy 
        "image_id": _bytes_feature(image_id),
        "image": _bytes_feature(_image2bytes(image)),
        "label": _int64_feature(label)
    }
    
    #Sve kreirane značajke dodaj u Features strukturu
    features = tf.train.Features(feature=feature_dict)
    
    #Zatim Features strukturu dodaju u Example
    example = tf.train.Example(features=features)
    
    return example.SerializeToString()

def video2example(video_id, image_seq, labels, num_images, num_labels):
    '''Radi pretvorbu video_id-a, niza sličica i oznaka iz videa u 
    serijalizirani SequenceExample'''
    
    #Kreiramo riječnik za metapodatke o videu i oznakama- kontrolni podatci
    context_dict = {
        "video_id": _bytes_feature(video_id),
        "num_images": _int64_feature(num_images),
        "num_labels": _int64_feature(num_labels)
    }
    #Kreiramo riječnik za niz slika i oznaka
    seq_dict = {
        "image_seq": _image_seq2FeatureList(image_seq),
        "labels": _labels2FeatureList(labels)
    }
    #Kreiramo Features za metapodatke
    context_feats = tf.train.Features(feature=context_dict)

    #Kreiramo FeatureLists kao zajedničku strukturu za niz slika i oznaka 
    sequence_lists = tf.train.FeatureLists(feature_list=seq_dict)
    
    #Kreiramo SequenceExample
    example = tf.train.SequenceExample(context=context_feats, 
                                       feature_lists=sequence_lists)
    
    return example.SerializeToString()


#%%Funkcije za parsiranje slika i videa iz TFRecord formata

#Funkcija za parsiranje Example-ova potpuno spremnih za obradu ResNet50 mrežom
def example2image_resnet(feat_extraction=True):
    '''Funkcija za parsiranje Example-a potpuno spremnih za obradu ResNet50 mrežom,
    definirana kao factory funkcija, kako bi bilo moguće definirati da li 
    parser koristimo za  izvlačenje značajki ili učenje modela.''' 
    
    def parser(example):
        '''Funkcija radilica koja obavlja parsiranje.'''
    
        #Obavezno definiramo opis značajki u Example-u, (shape, dtype)
        feat_desc = {
            "image_id": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }
        #Za mini grupu opažanja, napravimo parsiranje
        sample = tf.io.parse_example(example, feat_desc)
            
        ##Izvlačenje komponenti opažanja##
        #Ovu komponentu uključiti samo kod izvlačenja značajki, kod treniranja i testiranja
        #isključiti jer model očekuje samo 2 komponente
        image_id = sample['image_id']
            
        #Dekodiranje raw byte stringa u sliku
        image = tf.io.decode_raw(sample['image'], tf.float32)
        #Vraćanje u odgovarajuće dimenzije
        #Ako je u pipeline-u batch prije map-a dodaj -1 (zbog dimenzije batch-a), te raspakiraj dimenzije slike
        image = tf.reshape(image, [-1, 224, 224, 3])
            
        #Izvlačenje oznaka
        label = tf.cast(sample['label'], dtype=tf.int32)
    
        #Da li je parser korišten kod učenja modela
        if feat_extraction:
            return image_id, image, label
        else:
            return image, label
    
    return parser
    
#Funkcija za parsiranje Example-ova čije sličice imaju samo izmjenjenu veličinu
#te je dodatno potrebno obraditi da bi bili spremni za obradu ResNet50 mrežom
def example2sample_resized(feat_extraction=True, resnet_prep=True):
    '''Funkcija za parsiranje Example-a kojeg treba dodatno pripremiti
    za obradu ResNet50 mrežom, definirana kao factory funkcija,
    kako bi bilo moguće definirati da li parser koristimo za 
    izvlačenje značajki ili učenje modela te da li je potrebno raditi ResNet50
    pripremu podataka.'''
    
    def parser(example):
        '''Funkcija radilica koja obavlja parsiranje.'''

        #Obavezno definiramo opis značajki u Exampleu, (shape, dtype)
        feat_desc = {
            "image_id": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }
        #Za mini grupu opažanja, napravimo parsiranje
        sample = tf.io.parse_example(example, feat_desc)
            
        ##Izvlačenje komponenti opažanja##
        #Ovu komponentu vratiti samo prilikom izvlačenja značajki!!, 
        #kod treniranja i testiranja isključiti jer model očekuje samo 2 komponente
        image_id = sample['image_id']
            
        #Dekodiranje raw byte stringa u sliku
        image = tf.io.decode_raw(sample['image'], tf.uint8)   
        #Vraćanje u odgovarajuće dimenzije
        #Ako je u pipeline-u batch prije map-a dodaj -1 (zbog dimenzije batch-a), te raspakiraj dimenzije slike
        image = tf.reshape(image, [-1, 224, 224, 3])
        image = tf.cast(image, tf.float32)
        
        if resnet_prep:
            #ResNet preprocesing postupak (centriraj, obrni iz RGB u BGR redoslijed kanala)
            imagenet_channel_mean = tf.constant([123.68, 116.779, 103.939], dtype = tf.float32)
            image = image - imagenet_channel_mean
            image = image[:,:,:,::-1]
        else:
            image = tf.divide(image, tf.constant(255., tf.float32)) 
            
        #Izvlačenje oznaka
        label = tf.cast(sample['label'], dtype=tf.int32)
        
        #Da li je parser korišten kod učenja modela
        if feat_extraction:
            return image_id, image, label
        else:
            return image, label
     
    return parser
    
def example2video(feature_dim=2048, training=True):
    '''Funkcija za parsiranje SequenceExample-a, definirana kao factory funkcija,
    kako bi bilo moguće definirati dimenzionalnost sličica u videu te da li parser koristimo
    kod učenja.'''

    def parser(seq_example):
        '''Funkcija radilica koja obavlja parsiranje.'''
        #Obavezno definiramo opis značajki u Exampleu, (shape, dtype)
        #Context feats
        context_feat = {
            "video_id": tf.io.FixedLenFeature([], tf.string),
            "num_images": tf.io.FixedLenFeature([], tf.int64),
            "num_labels": tf.io.FixedLenFeature([], tf.int64)
        }
        #Seq feats
        seq_feat = {
            "image_seq": tf.io.FixedLenSequenceFeature([], tf.string),
            "labels": tf.io.FixedLenSequenceFeature([], tf.int64)
        }

        #Za mini grupu opažanja, napravimo parsiranje, ne zanimaju nas duljine
        context, sample, _ = tf.io.parse_sequence_example(seq_example, context_features=context_feat,
                                            sequence_features=seq_feat)
        
        #Context značajke
        #Ove komponente uključiti samo kod testiranja parsera, kod treniranja i testiranja
        #isključiti jer model očekuje samo 2 komponente
        video_id = context["video_id"]
        num_images = context["num_images"]
        num_labels = context["num_labels"]

        #Dekodiranje raw byte stringa u sliku i vraćanje u odgovarajuće dimenzije (ne marimo za broj opažanja)
        image_seq = tf.io.decode_raw(sample["image_seq"], tf.float32)
        #Ovdje je -1, za vremensku dimenziju, NE ZA BATCH!! Radimo na jednom opažanju
        image_seq = tf.reshape(image_seq, [-1, feature_dim])
        
        #Izvlačenje oznaka - mora biti int, zbog internog one hot kodiranja
        labels = tf.cast(sample["labels"], dtype=tf.int32)
        
        #Da li je parser korišten kod učenja modela
        if training:
            return image_seq, labels
        else:
            num_images = tf.cast(num_images, tf.int32)
            num_labels = tf.cast(num_labels, tf.int32)
            return video_id, num_images, num_labels, image_seq, labels
    
    return parser