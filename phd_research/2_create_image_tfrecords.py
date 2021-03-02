#PRIMJENA
'''
Spremanje sličica pripremljenih za obradu ResNet50 modelom u TFRecord formatu

Dva načina pripreme:
    1) samo se mijenja dimenzija ulaznih sličica u 224x224 te one ostaju u uint8 
    formatu;
    2) mijenja se dimenzija sličice, centriranje po kanalima i zamjena redoslijeda 
    kanala uslijed čega su sličice u float32 formatu.
    
Prvi način priprema ima manji trag na disku i omogućava korištenja Colab virtualnih
instanci. Drugi način pripreme rezultira većim opterečenjem diska (cca 4x većim),
ali omogućava nešto brži ulazni pipeline (pripremu mini grupa).

Izlaz su pripremljene sličice prvim i drugim načinom spremljene u TFRecord formatu
'''
#%%Biblioteke
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import read_labels, image2example
import os
import tensorflow as tf
import numpy as np

#%%Spremanje u TFRecord sličica pripremljenih 1. načinom

#Bazni direktoriji sa lokacijom ulaznih slika i oznaka te odredištem TFRecorda
base_image_dir = config.IMAGES_RESIZED
base_label_dir = config.LABELS
base_tfrecord_dir = config.TFR_IMG_RESIZED

#Kadrovi i podjele podataka
kadrovi = ["HE", "Fokus"]
data_splits = ["train", "val", "test"]

#Petlja po kadrovima
for kadar in kadrovi:
    
    #Inicijalizacija brojača sličica u kadru
    num_images = 0
    #Inicijalizacija brojača uzoraka u kadru
    num_samples = 0
    
    #Petlja po podjelama podataka
    for data_split in data_splits:
        
        #Trenutne putanje do sličica
        image_path = os.path.join(base_image_dir, kadar, data_split)
        
        #Petlja po video_id-u
        for video_id in os.listdir(image_path):
            
            #Putanja do trenutnog videa
            video_path = os.path.join(image_path, video_id)
            
            #Popis sličica u videu, potrebno ga je SORTIRATI, tako da se poklapa sa oznakama!!!
            image_names = sorted(os.listdir(video_path))
            
            #Lista oznaka trenutnog videa
            label_path = os.path.join(base_label_dir, data_split, video_id)
            labels = read_labels(label_path)
            
            #Kreiranje direktorija za izlazne TFRecord
            dest_dir = os.path.join(base_tfrecord_dir, kadar, data_split)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            #Definiranje izlazne TF record datoteke
            dest_file = os.path.join(dest_dir, video_id)
            
            #Otvaranje TF pisača, sve sličice i oznake jednog videa idu u isti TFRecord
            with tf.io.TFRecordWriter(dest_file + '.tfrecord') as writer:
                
                #Petlja po sličicama i njihovim oznakama
                for image_name, label in zip(image_names, labels):
                    
                    #Elementi Example-a
                    #Enkodiraj u byte image_id string
                    image_id = os.path.join(kadar, data_split, video_id, image_name.split(".")[0]).encode()
                    image = np.load(os.path.join(video_path, image_name))
                    
                    #Kreiranje serijaliziranog Example-a
                    example = image2example(image_id, image, label)
                    #Zapiši ga 
                    writer.write(example)
                    
                    #Ažuriraj brojač sličica
                    num_images += 1
            
            #Ažuriraj brojač opažanja
            num_samples += 1
    
    #Ispiši informativnu poruku
    print(f"[INFO] obrađen je kadar {kadar}, koji sadrži {num_samples} video zapisa i {num_images} sličica.")
    
#%%Spremanje u TFRecord sličica pripremljenih 2. načinom

#Bazni direktoriji sa lokacijom ulaznih slika i oznaka te odredištem TFRecorda
base_image_dir = config.IMAGES_RESNET
base_label_dir = config.LABELS
base_tfrecord_dir = config.TFR_IMG_RESNET

#Kadrovi i podjele podataka
kadrovi = ["HE", "Fokus"]
data_splits = ["train", "val", "test"]

#Petlja po kadrovima
for kadar in kadrovi:
    
    #Inicijalizacija brojača sličica u kadru
    num_images = 0
    #Inicijalizacija brojača uzoraka u kadru
    num_samples = 0
    
    #Petlja po podjelama podataka
    for data_split in data_splits:
        
        #Trenutne putanje do sličica
        image_path = os.path.join(base_image_dir, kadar, data_split)
        
        #Petlja po video_id-u
        for video_id in os.listdir(image_path):
            
            #Putanja do trenutnog videa
            video_path = os.path.join(image_path, video_id)
            
            #Popis sličica u videu, potrebno ga je SORTIRATI, tako da se poklapa sa oznakama!!!
            image_names = sorted(os.listdir(video_path))
            
            #Lista oznaka trenutnog videa
            label_path = os.path.join(base_label_dir, data_split, video_id)
            labels = read_labels(label_path)
            
            #Kreiranje direktorija za izlazne TFRecord
            dest_dir = os.path.join(base_tfrecord_dir, kadar, data_split)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            #Definiranje izlazne datoteke
            dest_file = os.path.join(dest_dir, video_id)
            
            #Otvaranje TF pisača, sve sličice i oznake jednog videa idu u
            #isti TFRecord
            with tf.io.TFRecordWriter(dest_file + '.tfrecord') as writer:
                
                #Petlja po sličicama i njihovim oznakama
                for image_name, label in zip(image_names, labels):
                    
                    #Elementi Example-a
                    #Enkodiraj u byte image_id string
                    image_id = os.path.join(kadar, data_split, video_id, image_name.split(".")[0]).encode()
                    image = np.load(os.path.join(video_path, image_name))
                    
                    #Kreiranje serijaliziranog Example-a
                    example = image2example(image_id, image, label)
                    #Zapiši ga 
                    writer.write(example)
                    
                    #Ažuriraj brojač sličica
                    num_images += 1
            
            #Ažuriraj brojač opažanja
            num_samples += 1
    
    #Ispiši informativnu poruku
    print(f"[INFO] obrađen je kadar {kadar}, koji sadrži {num_samples} video zapisa i {num_images} sličica.")