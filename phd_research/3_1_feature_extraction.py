#PRIMJENA
'''
Izvlačenje prostornih značajki primjenom predtrenirane ResNet50 mreže 
na ImageNet skupu podataka, za kadar HE i Fokus, za sličice 
pripremljene 1. načinom pripreme (samo resize na 224x224).

Izlaz su prostorne značajke dimenzija 2048 izvučene iz Average pooling sloja u
ndarray formatu tipa float32.
'''
#%%Biblioteka
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import example2sample_resized
import tensorflow as tf
import glob
from tqdm import tqdm
import os
import numpy as np

#%%Definiranje modela za izvlačenje značajki
tf.keras.backend.clear_session()

#Definiranje modela, sa završnim global average pooling slojem
bazni_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False,
                                             pooling="avg",
                                             input_tensor=tf.keras.layers.Input(shape=(224,224,3)))

#Model za izvlačenje značajki
feature_extraction_model = tf.keras.Model(bazni_model.input, bazni_model.output)

#Svi slojevi se zamrzavaju (ovo čak i nije potrebno u konkretnom slučaju)
feature_extraction_model.trainable = False

#%%Definiranje jednostavnog Dataset objekta za ulazne TFRecord-e
#Generiranje liste putanja do TFRecorda pripremljenih 1. načinom

#Ulazni direktorij sa resized sličicama
ulazni_dir = config.TFR_IMG_RESIZED
filenames = glob.glob(os.path.join(ulazni_dir, "**", "*.tfrecord"), recursive=True)

#Bitno je da sličice idu po redu, zbog spremanja prema mojoj strukturi
feat_dataset = tf.data.TFRecordDataset(filenames)
feat_dataset= feat_dataset.batch(16)
feat_dataset = feat_dataset.map(example2sample_resized(feat_extraction=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
feat_dataset = feat_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#%%Petlja za izvlačenje značajki
#Bazni direktoriji sa odredišnom lokacijom značajki
base_feats_dir = config.VIDEO_FEATS_31

#Incijalizacija oznake videa koji je u obradi
current_video_id = None

#Inicijalizacija image_id koji je u obradi
current_image_id = None

#Inicijalizacija spremnika za izvučene značajke iz ISTOG VIDEA (isti video_id)!!!
current_video_features = []

#Petlja po id-ovima slika te slikama, oznake nas ne zanimaju kod izvlačenja značajki
#Id-slika sadrži u sebi i oznaku videa kojem pripadaju!!!
for image_ids, images, _ in tqdm(feat_dataset):
    
    #Izvlačnje značajki za grupu slika 
    batch_features = feature_extraction_model(images)
    #Osiguranje da su dimenzije izvučenih značajki (br.uzoraka x 2048)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1))
    
    #Petlja po id-ovima slika i izvučenim značajkama
    for image_id, features in zip(image_ids.numpy(), batch_features.numpy()):
        
        #Izvuci video_id iz image_id-a, ostalo nas ne zanima (image_id => kadar/data_split/video_id/image_name)
        _, _ , video_id, _ = image_id.decode().split(os.sep)  
        
        #U slučaju da se promijenio video iz čijih slika izvlačimo značajke,
        #trebamo zapisati sve značajke iz spremnika current_video_features
        if video_id != current_video_id and current_video_id is not None:
            
            #Potrebne komponente iz current_image_id-a
            kadar, data_split, _, _ = current_image_id.decode().split(os.sep)
            
            #Struktura izlaznog direktorija
            output_dir = os.path.join(base_feats_dir, kadar, data_split)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            #Izlazna datoteke za video prije promjene id-a
            output_file = os.path.join(output_dir, current_video_id + ".npy")
            np.save(output_file, current_video_features)
            
            #Bitan korak je nakon što zapišemo podatke, da ponovno inicijaliziramo spremnik
            current_video_features = []
        
        #Zapiši koji je video trenutno u obradi i sličica, te dodaj njegove značajke u spremnik
        current_video_id = video_id
        current_image_id = image_id
        current_video_features.append(features)

#Zapiši i zadnji video
kadar, data_split, _, _ = current_image_id.decode().split(os.sep)
output_dir = os.path.join(base_feats_dir, kadar, data_split)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, current_video_id + ".npy")
np.save(output_file, current_video_features)