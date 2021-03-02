#PRIMJENA
'''
Priprema video sličica za obradu ResNet50 mrežom.

Dva načina pripreme:
    1) samo se mijenja dimenzija ulaznih sličica u 224x224 te one ostaju u uint8 
    formatu;
    2) mijenja se dimenzija sličice, centriranje po kanalima i zamjena redoslijeda 
    kanala uslijed čega su sličice u float32 formatu.
    
Prvi način priprema ima manji trag na disku i omogućava korištenja Colab virtualnih
instanci. Drugi način pripreme rezultira većim opterečenjem diska (4x većim),
ali omogućava nešto brži ulazni pipeline (pripremu mini grupa).

Izlaz su pripremljene sličice prvim i drugim načinom.
'''
#%%Bibliotke
from phd_lib.data_pipeline.image_preprocessors import resize_img, preprocess_img_for_ResNet50
from phd_lib import config 
from tqdm import tqdm
import glob
import os
import numpy as np

#%%Definiranje ulaznih i izlaznih direktorija
#Ulazni direktorij sa slikama
ulazni_dir= config.INPUT_DATASET

#Generiranje liste putanja do svih slika
img_paths = glob.glob(os.path.join(ulazni_dir, "**", "*.jpeg"), recursive=True)

#%%1. način pripreme sličica (resize 224x224)
#Izlazni direktorij za RESIZED slike
izlazni_dir = config.IMAGES_RESIZED

#Petlja po putanjama do slika
for img_path in tqdm(img_paths):
    
    #Procesuiranje slike u np.array za obradu ResNet50 modelom
    img_array = resize_img(img_path)
    
    #Struktura direktorija za pohranu obrađene slike - reži na separatoru direktorija
    *_, kadar, podjela_podataka, video_id, _ = img_path.split(os.sep)
    
    #Odredišni direktorij
    dest_dir = os.path.join(izlazni_dir, kadar, podjela_podataka, video_id)
   
    #Provjeri da li postoji odredišni direktorij, ako ne kreiraj ga
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    #Naziv datoteke, izbaci .jpeg ekstenziju
    img_name = os.path.basename(img_path).split(".")[0]
    #Puna putanje za spremanje
    array_path = os.path.join(dest_dir, img_name)
    
    #Spremanje
    np.save(array_path, img_array)

#%%2. način pripreme sličice (full ResNet50 spremni podatci)
#Izlazni direktorij za RESNET spremne slike
izlazni_dir = config.IMAGES_RESNET

#Petlja po putanjama do slika
for img_path in tqdm(img_paths):
    
    #Procesuiranje slike u np.array za obradu ResNet50 modelom
    img_array = preprocess_img_for_ResNet50(img_path)
    
    #Struktura direktorija za pohranu obrađene slike
    *_, kadar, podjela_podataka, video_id, _ = img_path.split(os.sep)
    #Odredišni direktorij
    dest_dir = os.path.join(izlazni_dir, kadar, podjela_podataka, video_id)
   
    #Provjeri da li postoji odredišni direktorij, ako ne kreiraj ga
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    #Naziv datoteke, izbaci .jpeg ekstenziju
    img_name = os.path.basename(img_path).split(".")[0]
    #Puna putanje za spremanje
    array_path = os.path.join(dest_dir, img_name)
    
    #Spremanje
    np.save(array_path, img_array)