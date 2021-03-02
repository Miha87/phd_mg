#PRIMJENA
'''
Učenje odabranog tipa modela, na jednom od tri moguća tipa značajki za jednu
od tri vrste kadrova.

Tri tipa modela su:
    1) LSTM model
    2) Dvosmjerni LSTM model
    3) Konvolucijski više etapni model
    
Tri tipa ulaznih značajki su:
    1) feature_extraction - čisti izračun značajki predtreniranom ResNet50 mrežom
    b) transfer_learning- izračun značajki sa fino podešenom ResNet50 mrežom i dodanom klasifikacijskom glavom
    c) train_base - izračun značajki sa modelom temeljenim na rezidualnim blokovima potpuno
    naučenim na vlastitim podatcima

Tri vrste kadrova su:
    1) HE - pogled iznad glave operatera sa pregledom cijelog radnog prostora
    2) Fokus - pogled usmjeren na ruke operatera sa pregledom objekta rada
    3) concat - konkatenacija kadra 1) i 2)
 
Izlaz je naučen model prema definiranim postavkama.
'''
#%%Biblioteka
from phd_lib import config
from phd_lib.callbacks.lr_helpers.lr_finder import LearningRateFinder
from phd_lib.callbacks.lr_helpers.cyclical_learning_rate import CyclicLR
from phd_lib.callbacks.lr_helpers.learning_rate_schedulers import PolynomialDecay
from phd_lib.callbacks.training_monitor import TrainingMonitor, MultiOutTrainingMonitor, monitor_data_reader
from phd_lib.callbacks.epoch_checkpoint import EpochCheckpoint
from phd_lib.data_pipeline.pipe_builders import build_train_pipeline, build_test_pipeline
from phd_lib.data_pipeline.tfrecord_helpers import example2video
from phd_lib.models.lstm_model import build_lstm_model
from phd_lib.models.ms_tcn_model import build_ms_tcn_model, SegmentationLoss, MaskConv1D, PredictionGeneration, Refeinment, DilatedResidualModule

import tensorflow as tf
import glob
import os

#%%Konfiguracija skripte
#Tipovi modela "LSTM", "biLSTM", "CONV"
MODEL_TYPE = "CONV"

#Tipovi značajki "feat_extraction", "transfer_learning", "train_base"
FEAT_TYPE = "train_base"

#Kadrovi "HE", "Fokus", "concat"
KADAR = "HE" 

#Putanje za pohranu modela, rezultata metrike, vizualizacije krivulja
MODEL_PATH = os.path.join(config.MODELS, "glava_modeli", MODEL_TYPE, FEAT_TYPE, 
                          KADAR, "EX_32") #Za svaki eksperiment mijenja se oznaka EX-a
HISTORY_PATH = os.path.join(MODEL_PATH, "metrics.json")
PLOT_PATH = os.path.join(MODEL_PATH, "plot.png")

#Kreiranje direktorija za pohranu
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#Hiperparametri učenja
BATCH_SIZE = 16 if KADAR == "concat" else 32
INIT_LR = 1e-2
EPOCHS = 40
#Kada učenje počinje od početka obavezno stavi 0, ne 1!!!
START_EPOCH = 0

#%%Logiranje hiperparametara modela, batch_size, lr, epochs
with open(os.path.join(MODEL_PATH, "log_file.txt"), "w") as log:
    log.write(f"batch_size:{BATCH_SIZE}, learning_rate:{INIT_LR}, epochs:{EPOCHS}\n")
    log.write(f"model_type:{MODEL_TYPE}, opis: LSTM - 256 neurona, 1 FC sloj, reg = 0.0005\n")
    log.write("optimizer: SGD, detalji: momentum=0.9, , loss: CCE\n")
    log.write("################ ANALIZA EKSPERIMENTA ##########################\n")
    log.write("PERFORMANSE:5 s/epohi\n")
    log.write("ZAKLJUČAK: Nije ostvarena ciljana točnost od 95%.\
              \nNakon 80 epoha stagnira rast točnosti i pad gubitka na validacijskom skupu\
              \ndok gubitak na trening skupu jako sporo pada i točnost jako sporo raste.\
              \nRezultati ukazuju da je problem u varijanci i visokoj pristranosti.\
              \nUz 266k parametara, zapinjemo u 88%.")

#%%Definiranje ulaznog pipeline-a
print("[INFO] priprema ulaznog pipeline-a")

#Pomoćna funkcija za definiranje putanja do značajki odgovarajućeg tipa, kadra i podjele podataka
def get_feature_files(feat_type, kadar, data_split):
    feats = {"feat_extraction": config.TFR_VIDEO_FEATS_31,
             "transfer_learning": config.TFR_VIDEO_FEATS_32,
             "train_base": config.TFR_VIDEO_FEATS_33}
    
    ulazni_dir = feats[feat_type]
    filenames = glob.glob(os.path.join(ulazni_dir, 
                                       kadar, 
                                       data_split, 
                                       "*.tfrecord"), recursive=True)
    return filenames

#Lokacija ulaznih podataka - potrebno je napraviti učenje za kadar HE i Fokus
train_filenames = get_feature_files(FEAT_TYPE, KADAR, "train")
val_filenames = get_feature_files(FEAT_TYPE, KADAR, "val")

#Generiranje train i val dataseta, za konkatenaciju kadrova dimenzija značajki je 4096, za ostale kadrove 2048
feature_dim = {"HE": 2048, "Fokus": 2048, "concat": 4096}

#Oba dataset-a su nadopunjena kako bi mogli koristi batch size veći od 1
train_dataset = build_train_pipeline(train_filenames, 
                                     example2video(feature_dim=feature_dim[KADAR], training=True), 
                                     batch_size=BATCH_SIZE, padded_batch=True)

val_dataset = build_train_pipeline(val_filenames, 
                                   example2video(feature_dim=feature_dim[KADAR], training=True), 
                                   batch_size=BATCH_SIZE, padded_batch=True)

#%%Definiranje modela za učenje
print("[INFO] priprema modela")
tf.keras.backend.clear_session()
#Komponente modela
if MODEL_TYPE == "CONV":
    model = build_ms_tcn_model(input_shape=(None, feature_dim[KADAR]), num_layers_PG=5, R_stages=1, num_layers_R=5, filters=64, training=True, dropout_rate=0.5, shared_R=False)
else:
    model = build_lstm_model(input_shape=(None, feature_dim[KADAR]), mask_value=0., bidirect=False, num_lstm_layers=1, 
                             lstm_units=256, lstm_dropout=0.5, lstm_recurrent_dropout=0., 
                             num_fc_layers=1, fc_reg=0.005, fc_units=256, fc_dropout=0.5, 
                             num_classes=10, seed=42)
#%%Podizanje modela iz odgovarajuće epohe
if MODEL_TYPE == "CONV":
    custom_objects = {"PredictionGeneration":PredictionGeneration, "Refeinment": Refeinment, 
                  "MaskConv1D": MaskConv1D, "DilatedResidualModule": DilatedResidualModule}
    model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "epoch_40"), custom_objects=custom_objects)
    
else:
    model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "epoch_40"))

#%%Kompajliranje modela za fino podešavanja
optimizer = tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=0.9)#decay=5e-2
loss = tf.keras.losses.SparseCategoricalCrossentropy()
#loss = SegmentationLoss()
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

#%%Traženje optimalne stope učenja
lr_finder = LearningRateFinder(model, stop_factor=1.5, smoothing_factor=0.9)
lr_finder.find_lr(train_dataset, y=None, num_samples=480,
                  batch_size=BATCH_SIZE, epochs=5, min_lr=1e-5, max_lr=1)

#%%Definiranje alata za pohranu modela i nadzor učenja
#Checkpointer modela
cm = EpochCheckpoint(output_path=MODEL_PATH, every=40, start_at_epoch=START_EPOCH)

#Trening monitor 
#tm = TrainingMonitor(plot_path=PLOT_PATH, history_path=HISTORY_PATH, start_at_epoch=START_EPOCH)

#MultiOut trening monitor
tm = MultiOutTrainingMonitor(plot_path=PLOT_PATH, history_path=HISTORY_PATH, start_at_epoch=START_EPOCH)

#Raspored stope učenja - ciklička stop učenja
schedule = CyclicLR(base_lr=1e-4, max_lr=1e-2, step_size=60)

#Izmjena stope učenja u slučaju da zapnemo
#schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

#linear_schedule = PolynomialDecay(max_epochs=40, init_lr=INIT_LR, power=1.0)
#schedule = tf.keras.callbacks.LearningRateScheduler(linear_schedule)
#%%Učenje modela
print("[INFO] model je u procesu učenja")
model.fit(train_dataset, epochs=EPOCHS, callbacks=[cm, tm,schedule], validation_data=val_dataset)


