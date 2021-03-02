#PRIMJENA
'''
Spremanje izračunatih značajki putem različitih baznih mreža u TFRecord formatu.

Ulazne značajke za video zapise su u ndarray formatu. Značajke su pripremljene
jednim od ova tri načina:
    a) 3_1_feature_extraction - čisti izračun značajki predtreniranom ResNet50 mrežom
    b) 3_2_transfer_learning_base_model - izračun značajki sa fino podešenom ResNet50 mrežom
    c) 3_3_train_base_model - izračun značajki sa modelom temeljenim na rezidualnim blokovima potpuno
    naučenim na vlastitim podatcima
    
Izlaz su prostorne značajke dimenzija 2048 izračunte za kadar HE i Fokus te značajke
koje su konkatenacija značajki oba kadra (dimenzija 4096), za svaki od tri tipa modela, tj.
9 različitih tipova prostornih značajki.
'''
#%%Biblioteke
import tensorflow as tf
import os
import numpy as np
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import read_labels, video2example

#%% Zapisivanje serijaliziranih SequenceExample-a u TFRecord, za oba kadra

#Bazni direktoriji sa lokacijom ulaznih video značajki i oznaka te odredištem TFRecorda
#postoje tri tipa mogućih ulaznih značajki *_31, *_32, *_33
base_video_dir = config.VIDEO_FEATS_33
#Bazni direktorij sa lokacijom oznaka
base_label_dir = config.LABELS
#Bazni direktorij odredišta TFRecorda
#postoje tri tipa mogućih izlaznih značajki *_31, *_32, *_33
base_tfrecord_dir = config.TFR_VIDEO_FEATS_33

#Maksimalan broj opažanja u train, val i test TFRecordima, brojke
#temeljene na preporukama o veličini TFRecord-a
max_examples_per_split = {"train": 60, "val": 70, "test": 70}

#Kadrovi i podjele podataka
kadrovi = ["HE", "Fokus"]
data_splits = ["train", "val", "test"]

#Petlja po kadrovima
for kadar in kadrovi:
    
    #Petlja po podjelama podataka
    for data_split in data_splits:
        
        #Trenutna putanja
        current_path = os.path.join(base_video_dir, kadar, data_split)
        
        #Generiraj listu videa u odgovarajućem kadru i podjeli podataka
        video_list = os.listdir(current_path)
        
        #Maksimalan broj opažanja u jednom TFRecord-u
        max_examples = max_examples_per_split[data_split]
        
        #Izračunaj broj TFRecorda za data split
        num_records = int(np.ceil(len(video_list) / max_examples))

        #Generiraj grupe videa koje idu u isti TFRecord
        video_batches = [video_list[(idx * max_examples):((idx + 1) * max_examples)] for idx in range(num_records)]

        #Petlja po grupama videa
        for tfr_id, video_batch in enumerate(video_batches, start=1):
            
            #Kreiranje direktorija za izlazne TFRecorde
            dest_dir = os.path.join(base_tfrecord_dir, kadar, data_split)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            #Kreiranje naziva datoteke izlaznog TFRecorda
            dest_file = os.path.join(dest_dir, f"{tfr_id}_grupa.tfrecord")
        
            #Otvaranje TF pisača, više videa i oznaka ide u isti TFRecord
            with tf.io.TFRecordWriter(dest_file) as writer:
          
                #Petlja po video zapisima u grupi videa
                for video in video_batch:
                
                    #Izvuci video_id
                    video_id = video.split(".")[0] 
                
                    #Putanja do trenutnog videa
                    video_path = os.path.join(current_path, video)

                    #Putanja do oznaka trenutnog videa
                    labels_path = os.path.join(base_label_dir, data_split, video_id)
                
                    #Učitaj video i oznake
                    video_images = np.load(video_path)
                    labels = read_labels(labels_path)

                    #Izvuci informacije o broju slika i oznaka i potvrdi da su jednake
                    num_images = video_images.shape[0]
                    num_labels = len(labels)
                    assert num_images == num_labels, "Broj sličica u video zapisu i oznaka mora biti jednak!"
                
                    #Kreiraj serijalizirani SequenceExample
                    example = video2example(video_id.encode(), video_images, labels, num_images, num_labels)
                    #Zapiši ga
                    writer.write(example)
                    
            #Ispiši informativnu poruku
            print(f"[INFO] Zapisana je {tfr_id}. grupa za {kadar}/{data_split} podatke.")
            
#%% Zapisivanje serijaliziranih SequenceExample-a u TFRecord, za KONKATENACIJU oba kadra
            
#Bazni direktoriji sa lokacijom ulaznih video značajki i oznaka te odredištem TFRecorda
#postoje tri tipa mogućih ulaznih značajki *_31, *_32, *_33
base_video_dir = config.VIDEO_FEATS_33
#Bazni direktorij sa lokacijom oznaka
base_label_dir = config.LABELS
#Bazni direktorij odredišta TFRecorda
#postoje tri tipa mogućih izlaznih značajki *_31, *_32, *_33
base_tfrecord_dir = config.TFR_VIDEO_FEATS_33

#Maksimalan broj opažanja u train, val i test TFRecordima, brojke
#temeljene na preporukama o veličini TFRecord-a
max_examples_per_split = {"train": 30, "val": 35, "test": 35}

#Kadrovi i podjele podataka
kadrovi = ["HE", "Fokus"]
data_splits = ["train", "val", "test"]

#Petlja po podjelama podataka
for data_split in data_splits:
        
    #Trenutna putanja
    current_path_HE = os.path.join(base_video_dir, "HE", data_split)
    current_path_Fokus = os.path.join(base_video_dir, "Fokus", data_split)
    
    #Generiraj listu videa u odgovarajućoj podjeli podataka (ista je lista, neovisno od kadra)
    video_list = os.listdir(current_path_HE)
    
    #Maksimalan broj opažanja u jednom TFRecord-u
    max_examples = max_examples_per_split[data_split]
    
    #Izračunaj broj TFRecorda za data split
    num_records = int(np.ceil(len(video_list) / max_examples))

    #Generiraj grupe videa koje idu u isti TFRecord
    video_batches = [video_list[(idx * max_examples):((idx + 1) * max_examples)] for idx in range(num_records)]

    #Petlja po grupama videa
    for tfr_id, video_batch in enumerate(video_batches, start=1):
        
        #Kreiranje direktorija za izlazne TFRecorde
        dest_dir = os.path.join(base_tfrecord_dir, "concat", data_split)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        #Kreiranje naziva datoteke izlaznog TFRecorda
        dest_file = os.path.join(dest_dir, f"{tfr_id}_grupa.tfrecord")
    
        #Otvaranje TF pisača, više videa i oznaka ide u isti TFRecord
        with tf.io.TFRecordWriter(dest_file) as writer:
      
            #Petlja po video zapisima u grupi videa
            for video in video_batch:
            
                #Izvuci video_id
                video_id = video.split(".")[0] 
            
                #Putanja do trenutnog videa iz oba kadra
                video_path_HE = os.path.join(current_path_HE, video)
                video_path_Fokus = os.path.join(current_path_Fokus, video)

                #Putanja do oznaka trenutnog videa
                labels_path = os.path.join(base_label_dir, data_split, video_id)
            
                #Učitaj videe za oba kadra i oznake
                video_images_HE = np.load(video_path_HE)
                video_images_Fokus = np.load(video_path_Fokus)
                labels = read_labels(labels_path)
                
                #Konkateniraj značajke videa po vremenskoj osi - svaki vremenski korak sada ima dim 4096
                concat_video_images = np.concatenate((video_images_HE, video_images_Fokus), axis=1)

                #Izvuci informacije o broju slika i oznaka i potvrdi da su jednake
                num_images = video_images_HE.shape[0]
                num_labels = len(labels)
                assert num_images == num_labels, "Broj sličica u video zapisu i oznaka mora biti jednak!"
            
                #Kreiraj serijalizirani SequenceExample
                example = video2example(video_id.encode(), concat_video_images, labels, num_images, num_labels)
                #Zapiši ga
                writer.write(example)
                
        #Ispiši informativnu poruku
        print(f"[INFO] Zapisana je {tfr_id}. grupa za {data_split} podatke.")
