#PRIMJENA
'''
Izvlačenje prostornih značajki primjenom modela sa rezidualnim blokovima
u potpunosti naučenog na pojedinačnim sličicama posebno za kadar HE i Fokus. 
(za sličice  pripremljene 1. načinom pripreme (samo resize na 224x224).
                   
Koraci:
    1) Definiranje arhitekture i hiperparametara novog modela sa rezidualnim
    blokovima.
    2) Učenje modela iz definiranog u koraku 1) za svaki kadar zasebno tj.
    radimo 2 modela.
    3) Po učenju izvlačimo značajke iz zadnjeg konvloucijskog sloja
    definiranog u koraku 1). 

Izlaz su prostorne značajke dimenzija 2048 izvučene iz posljednjeg konvolucijskog
sloja u ndarray formatu tipa float32.
'''
#%%Biblioteke
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import example2sample_resized
from phd_lib.data_pipeline.pipe_builders import build_train_pipeline, build_test_pipeline
from phd_lib.callbacks.training_monitor import TrainingMonitor
from phd_lib.callbacks.epoch_checkpoint import EpochCheckpoint
from phd_lib.models.resnet_model import build_resnet_model
import tensorflow as tf
from sklearn.metrics import classification_report
import glob
import numpy as np
import os
from tqdm import tqdm

#%%Konfiguracija skripte
KADAR = "HE" #Nakon toga odradi ovo i za Fokus
BATCH_SIZE = 32
INIT_LR = 1e-5
EPOCHS = 4
#Kada učenje počinje od početka obavezno stavi 0, ne 1!!!
START_EPOCH = 4
MODEL_PATH = os.path.join(config.MODELS, "bazni_modeli", "train_base", 
                          KADAR, "EX_2") #Za svaki eksperiment mijenja oznaku EX-a
HISTORY_PATH = os.path.join(MODEL_PATH, "metrics.json")
PLOT_PATH = os.path.join(MODEL_PATH, "plot.png")

#%%Kreiranje direktorija
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#%%Logiranje hiperparametara modela, batch_size, lr, epochs
with open(os.path.join(MODEL_PATH, "log_file.txt"), "w") as log:
    log.write("Broj filter (64, 256, 512, 1024, 2048), broj etapa (1, 1, 1, 1), batch_size=32, reg=0.0005, epochs=4, SGD, LR 1e-1, momentum=0.9\n")
    log.write("Nakon 2 epohe spuštamo LR na 1e-3\n")
    log.write("Nakon 4 epohe spuštamo LR na 1e-5")
  
#%%Definiranje ulaznog pipeline-a
print("[INFO] priprema ulaznog pipeline-a")

#Pomoćna funkcija za definiranje putanja do slika iz odgovarajućeg kadra i podjele podataka
def get_img_files(kadar, data_split):
    ulazni_dir = config.TFR_IMG_RESIZED
    filenames = glob.glob(os.path.join(ulazni_dir, 
                                       kadar, 
                                       data_split, 
                                       "*.tfrecord"), recursive=True)
    return filenames

#Lokacija ulaznih podataka - potrebno je napraviti učenje za kadar HE i Fokus
train_filenames = get_img_files(KADAR, "train")
val_filenames = get_img_files(KADAR, "val")

#Generiranje train i val dataseta - samo normalizacija podataka => resnet_prep=False
train_dataset = build_train_pipeline(train_filenames, example2sample_resized(feat_extraction=False, resnet_prep=False), batch_size=BATCH_SIZE)
val_dataset = build_train_pipeline(val_filenames, example2sample_resized(feat_extraction=False, resnet_prep=False), batch_size=BATCH_SIZE)

#%%Definiranje modela za učenje
print("[INFO] priprema modela")
tf.keras.backend.clear_session()

#Definiranje baznog modela, bez klasifikacijskih slojeva
model = build_resnet_model(224, 224, 3, (1, 1, 1, 1), (64, 256, 512, 1024, 2048), num_classes=10, reduce_in_dim=True, reg=0.0005)

#%%
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "epoch_4"))

#%%Kompajliranje modela za fino podešavanja
#Optimizacijski postupaka uz definiran momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=0.9)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

#%%Definiranje alata za pohranu modela i nadzor učenja
#Checkpointer modela svake 4 etape
cm = EpochCheckpoint(output_path=MODEL_PATH, every=1, start_at_epoch=START_EPOCH)
#Trening monitor 
tm = TrainingMonitor(plot_path=PLOT_PATH, history_path=HISTORY_PATH, start_at_epoch=START_EPOCH)

#%%Učenje modela
print("[INFO] model je u procesu učenja")
model.fit(train_dataset, epochs=EPOCHS, callbacks=[tm, cm], validation_data=val_dataset)

#%%Evaluacija modela
print("[INFO] ocjena točnosti modela")
#Generiranja naziva aktivnosti
nazivi_aktivnosti = config.ACT_KLASE

#Generiranje skupova za ocjenu modela
data_splits = [train_filenames, val_filenames]

#Petlja po podjelama podataka
for split in data_splits:
    ds = build_test_pipeline(split, example2sample_resized(feat_extraction=False, resnet_prep=False), batch_size=BATCH_SIZE)
    
    y_true = [data_point[1] for data_point in ds.as_numpy_iterator()]
    y_true = np.hstack(y_true)
    
    y_pred = model.predict(ds)
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred, target_names=nazivi_aktivnosti))
    
#%%Izvlačenje značajki iz fino podešenih modela
#model_HE_4 -> model za kadar HE izvlačenje značajki iz prvog FC sloja, epoha 4, eksperiment 2
model_HE_4 = tf.keras.models.load_model(os.path.join(config.MODELS, "bazni_modeli","train_base", 
                          "HE", "EX_2", "epoch_4"))

#model_Fokus_5 -> model za kadar Fokus izvlačenje značajki iz prvog FC sloja, epoha 5, eksperiment 2
model_Fokus_5 = tf.keras.models.load_model(os.path.join(config.MODELS, "bazni_modeli","train_base", 
                          "Fokus", "EX_2", "epoch_5"))

#%%Ulaz i izlaz iz modela je aktivacija zadnjeg konvolucijskog bloka
#HE kadar
HE_baza = tf.keras.Model(inputs=model_HE_4.input, outputs=model_HE_4.layers[53].output)
output = tf.keras.layers.GlobalAveragePooling2D()(HE_baza.output)
feat_extraction_HE = tf.keras.Model(inputs=HE_baza.input,outputs=output)

#Fokus kadar
Fokus_baza = tf.keras.Model(inputs=model_Fokus_5.input, outputs=model_Fokus_5.layers[53].output)
output = tf.keras.layers.GlobalAveragePooling2D()(Fokus_baza.output)
feat_extraction_Fokus = tf.keras.Model(inputs=Fokus_baza.input, outputs=output)

#%%Zamrzavanje slojeva
feat_extraction_HE.trainable = False
feat_extraction_Fokus.trainable = False

#%%Pomoćna funkcija za generiranje dataset objekta za izvlačenje značajki za izabrani kadar
def generate_dataset(kadar, batch_size=16):  
    
    #Ulazni direktorij sa resized sličicama
    ulazni_dir = config.TFR_IMG_RESIZED
    filenames = glob.glob(os.path.join(ulazni_dir, kadar, "**", "*.tfrecord"), recursive=True)
    
    #Bitno je da sličice idu po redu, zbog spremanja prema mojoj strukturi
    feat_dataset = tf.data.TFRecordDataset(filenames)
    feat_dataset = feat_dataset.batch(batch_size)
    #Nije potrebna priprema za ResNet
    feat_dataset = feat_dataset.map(example2sample_resized(feat_extraction=True, resnet_prep=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    feat_dataset = feat_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return feat_dataset

#%%Pomoćna funkcija sa petljom za izvlačenje značajki
def extract_feat(model, dataset, base_feats_dir=config.VIDEO_FEATS_33):
    
    #Bazni direktoriji sa odredišnom lokacijom značajki - transfer learning
    base_feats_dir = base_feats_dir
    
    #Incijalizacija oznake videa koji je u obradi
    current_video_id = None
    
    #Inicijalizacija image_id koji je u obradi
    current_image_id = None
    
    #Inicijalizacija spremnika za izvučene značajke iz ISTOG VIDEA (isti video_id)!!!
    current_video_features = []
    
    #Petlja po id-ovima slika te slikama, oznake nas ne zanimaju kod izvlačenja značajki
    #Id-slika sadrži u sebi i oznaku videa kojem pripadaju!!!
    for image_ids, images, _ in tqdm(dataset):
        
        #Izvlačnje značajki za grupu slika 
        batch_features = model(images)
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
    
    return None

#%%Izvlačenje značajki za oba kadra i modela
kadrovi = ["HE", "Fokus"]
modeli = [feat_extraction_HE, feat_extraction_Fokus]

for kadar, model in zip(kadrovi, modeli):
    dataset = generate_dataset(kadar, batch_size=16)
    extract_feat(model, dataset)
