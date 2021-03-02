
#%%Biblioteke 
import tensorflow as tf
from phd_lib.data_pipeline.pipe_builders import build_test_pipeline
from phd_lib.data_pipeline.tfrecord_helpers import example2video
from phd_lib.models.test_models import SegmentationLoss_v2
from phd_lib.models.ms_tcn_model import build_ms_tcn_model
from tensorflow.keras.utils import to_categorical
import os 
import glob

#%%Podatci
#Putanja do opažanja
path = r"D:\Phd_data\Video_feats_tfrecords\feat_extraction\HE\train"
train_tfs = glob.glob(os.path.join(path, "*.tfrecord"))

#%%Dva batch-a sa dva opažanja po batch-u
ds_no_pad = build_test_pipeline(train_tfs, example2video(2048), 2).take(2)
ds_padded = build_test_pipeline(train_tfs, example2video(2048), 2, padded_shapes=([457,None], [457])).take(2)

for element in ds_no_pad.take(2): 
    pred_1 = element[0]
    y_1 = element[1]
for element in ds_padded.take(2): 
    pred_2 = element[0]
    y_2 = element[1]

y_1_oh = to_categorical(y_1)
y_2_oh = to_categorical(y_2)
#%%Izgradi 2 modela, sa i bez maskiranja SA JEDNIM IZLAZOM i kompaliraj ih
base_model = build_ms_tcn_model(training=True)
base_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
loss_model = build_ms_tcn_model(training=True)
loss_model.compile(loss=SegmentationLoss_v2(), metrics=["accuracy"])

#%%Model predictions
p1_base = base_model(pred_1)
p2_base = base_model(pred_2)
p1_loss = loss_model(pred_1)
p2_loss = loss_model(pred_2)

#%%Evaluacija rezultata modela - sada radi ako ima i više izlaza samo sa jednom definicijom target podataka umjesto tuple-a
base_model.evaluate(pred_1, y_1)#(y_1, y_1, y_1, y_1))
base_model.evaluate(pred_2, y_2)#(y_2, y_2, y_2, y_2))
loss_model.evaluate(pred_1, (y_1, y_1, y_1, y_1))
loss_model.evaluate(pred_2, y_2) #(y_2, y_2, y_2, y_2))

#%%Fitanje modela
#base_model.fit(ds_no_pad, epochs=1)
#base_model.fit(ds_padded, epochs=1)
#loss_model.fit(ds_no_pad, epochs=1)
loss_model.fit(ds_padded, epochs=1)

#%%BIBLIOTEKE
import tensorflow as tf
import glob
import os
from phd_lib.data_pipeline.pipe_builders import build_train_pipeline, build_test_pipeline
from phd_lib.data_pipeline.tfrecord_helpers import example2sample_resized, example2video
from tensorflow.keras.models import load_model
from phd_lib import config

#%%Putanje do modela
#val_acc = 90,79
path_HE_tl = r"D:\Phd_data\Models\bazni_modeli\transfer_learning\HE\EX_1\epoch_15"
#val_acc = 95,11
path_Fokus_tl = r"D:\Phd_data\Models\bazni_modeli\transfer_learning\Fokus\EX_1\epoch_12"
#val_acc = 95,284
path_HE_base = r"D:\Phd_data\Models\bazni_modeli\train_base\HE\EX_2\epoch_4"
#val_acc = 95,48
path_Fokus_base = r"D:\Phd_data\Models\bazni_modeli\train_base\Fokus\EX_2\epoch_5"

#%%Putanje do ulaznih podataka
HE_data_val = glob.glob(os.path.join(config.TFR_IMG_RESIZED, "HE", "val","*.tfrecord"))
Fokus_data_val = glob.glob(os.path.join(config.TFR_IMG_RESIZED, "Fokus", "val","*.tfrecord"))

#Definiranje datasetova
HE_tl_dataset = build_train_pipeline(HE_data_val, example2sample_resized(feat_extraction=False, resnet_prep=True), 
                                   batch_size=32)
Fokus_tl_dataset = build_train_pipeline(Fokus_data_val, example2sample_resized(feat_extraction=False, resnet_prep=True), 
                                   batch_size=32)
HE_base_dataset = build_train_pipeline(HE_data_val, example2sample_resized(feat_extraction=False, resnet_prep=False), 
                                   batch_size=32)
Fokus_base_dataset = build_train_pipeline(Fokus_data_val, example2sample_resized(feat_extraction=False, resnet_prep=False), 
                                   batch_size=32)

#%%Definiranje modela
model_HE_tl = load_model(path_HE_tl)
model_Fokus_tl = load_model(path_Fokus_tl)
model_HE_base = load_model(path_HE_base)
model_Fokus_base = load_model(path_Fokus_base)

#%%Kompajliranje modela
model_HE_tl.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model_Fokus_tl.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model_HE_base.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model_Fokus_base.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

#%%Evaluacija modela
model_HE_tl.evaluate(HE_tl_dataset)
model_Fokus_tl.evaluate(Fokus_tl_dataset)
model_HE_base.evaluate(HE_base_dataset)
model_Fokus_base.evaluate(Fokus_base_dataset)

#%%Testiranje izlaza TL modela
#%%Test opažanja za HE i Fokus
HE_train_files =glob.glob(os.path.join(config.TFR_IMG_RESIZED,"HE", "train", "*.tfrecord"))
Fokus_train_files =glob.glob(os.path.join(config.TFR_IMG_RESIZED,"Fokus", "train", "*.tfrecord"))

# Opažanje 101_D1_V7_O1_T3\\slika-0001
he_train_data = build_test_pipeline(HE_train_files, example2sample_resized(feat_extraction=True, resnet_prep=True), batch_size=1)
fokus_train_data = build_test_pipeline(Fokus_train_files, example2sample_resized(feat_extraction=True, resnet_prep=True), batch_size=1)


for element in he_train_data.take(1): 
    he_test = element[1].numpy()

for element in fokus_train_data.take(1): 
    fokus_test = element[1].numpy()

   
#%%Izlazi iz tl modela od prvog potpuno povezanog sloja
tf.keras.backend.clear_session()
outputs_HE_tl = [layer.output for layer in model_HE_tl.layers[177:]]
outputs_Fokus_tl = [layer.output for layer in model_Fokus_tl.layers[177:]]

#Testni model
HE_multi = tf.keras.Model(inputs=model_HE_tl.input, outputs=outputs_HE_tl)
Fokus_multi = tf.keras.Model(inputs=model_Fokus_tl.input, outputs=outputs_Fokus_tl)

#Model kod transfer learninga
feat_extraction_HE = tf.keras.Model(model_HE_tl.input, model_HE_tl.layers[177].output)
feat_extraction_Fokus = tf.keras.Model(model_Fokus_tl.input, model_Fokus_tl.layers[177].output)

#%%Test opažanje

o1 = HE_multi(he_test)
o2 = feat_extraction_HE(he_test)


#Da li su izlazi isti #suma(83.139656)
print("Svi izlazi modela su isti:", (o1[0].numpy() == o2.numpy()).all())

o1 = Fokus_multi(fokus_test)
o2 = feat_extraction_Fokus(fokus_test)

#Da li su izlazi isti # suma(172.86882) 
print("Svi izlazi modela su isti:", (o1[0].numpy() == o2.numpy()).all())

#%%Test za train_base
HE_train_files =glob.glob(os.path.join(config.TFR_IMG_RESIZED,"HE", "train", "*.tfrecord"))
Fokus_train_files =glob.glob(os.path.join(config.TFR_IMG_RESIZED,"Fokus", "train", "*.tfrecord"))

# Opažanje 101_D1_V7_O1_T3\\slika-0001
he_train_data = build_test_pipeline(HE_train_files, example2sample_resized(feat_extraction=True, resnet_prep=False), batch_size=1)
fokus_train_data = build_test_pipeline(Fokus_train_files, example2sample_resized(feat_extraction=True, resnet_prep=False), batch_size=1)

for element in he_train_data.take(1): 
    he_test = element[1].numpy()

for element in fokus_train_data.take(1): 
    fokus_test = element[1].numpy()


#%%HE kadar
HE_multi = tf.keras.Model(inputs=model_HE_base.input, outputs=model_HE_base.layers[54].output)
Fokus_multi = tf.keras.Model(inputs=model_Fokus_base.input, outputs=model_Fokus_base.layers[54].output)

#HE kadar
HE_baza = tf.keras.Model(inputs=model_HE_base.input, outputs=model_HE_base.layers[53].output)
output = tf.keras.layers.GlobalAveragePooling2D()(HE_baza.output)
feat_extraction_HE = tf.keras.Model(inputs=HE_baza.input,outputs=output)

#Fokus kadar
Fokus_baza = tf.keras.Model(inputs=model_Fokus_base.input, outputs=model_Fokus_base.layers[53].output)
output = tf.keras.layers.GlobalAveragePooling2D()(Fokus_baza.output)
feat_extraction_Fokus = tf.keras.Model(inputs=Fokus_baza.input, outputs=output)

#%%test
o1 = HE_multi(he_test)
o2 = feat_extraction_HE(he_test)

#Da li su izlazi isti #suma(190.08653)
print("Svi izlazi modela su isti:", (o1[0].numpy() == o2.numpy()).all())


o1 = Fokus_multi(fokus_test)
o2 = feat_extraction_Fokus(fokus_test)

#Da li su izlazi isti # suma(302.64795) 
print("Svi izlazi modela su isti:", (o1[0].numpy() == o2.numpy()).all())


#%%Testiranje klasifikacije modela
import os
import glob
import numpy as np
from phd_lib import config
from phd_lib.data_pipeline.pipe_builders import build_test_pipeline
from phd_lib.models.ms_tcn_model import MaskConv1D, DilatedResidualModule, PredictionGeneration, Refeinment 
from phd_lib.data_pipeline.tfrecord_helpers import example2video
from phd_lib.metrics import mAP_at_IoU, f1_at_IoU, data_for_mAP
import tensorflow as tf
from tensorflow.keras.models import load_model

#%%Putanje do testnog modela
test_model_path = r"D:\Phd_data\Models\glava_modeli\CONV\train_base\Fokus\EX_1\epoch_40"

#Priprema ulaznog modela
custom_objects = {"MaskConv1D": MaskConv1D, "DilatedResidualModule": DilatedResidualModule, 
                      "PredictionGeneration": PredictionGeneration, "Refeinment": Refeinment}
model = load_model(test_model_path, custom_objects=custom_objects) 
#Samo izlaz iz zadnjeg sloja
outputs=[layer.output for layer in model.layers]
model = tf.keras.Model(inputs=model.input, outputs=outputs[-1])

#Kompajliranje modela
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
#%%Putanje do podataka
train_path = r"D:\Phd_data\Video_feats_tfrecords\train_base\Fokus\train"
val_path = r"D:\Phd_data\Video_feats_tfrecords\train_base\Fokus\val"

full_train_path = glob.glob(os.path.join(train_path, "*.tfrecord"))
full_val_path = glob.glob(os.path.join(val_path, "*.tfrecord"))

#%%
BATCH_SIZE=32
FEATURE_DIM = 2048
base_feats_dir = r"D:\Phd_data\Video_feats_tfrecords\train_base\Fokus"
def extract_pred_data(model, data_split):
    #Putanje do značajki za odgovarajuću podjelu podataka
    feat_path = glob.glob(os.path.sep.join([base_feats_dir, data_split, "*tfrecord"]))
    
    #Definiranje dataset-a, broj vremenskih koraka nadopunjuje se do najvećeg broja koraka u cijelom skupu
    #kako bi kod izrade grafa imali optimiziranu implementaciju
    #Struktura izlaza dataset-a: video_id, num_images, num_labels, image_seq, labels 
    dataset = build_test_pipeline(feat_path, example2video(feature_dim=FEATURE_DIM, training=False),
                                  batch_size=BATCH_SIZE, padded_batch=True, 
                                  padded_shapes=([], [], [], [457, None], [457]), 
                                  padding_values=("", 0, 0, 0.,0))
    
    #Izvlačenja duljina nizova bez nadopuna
    y_lens = tf.concat([data_point[1] for data_point in dataset.as_numpy_iterator()], axis=0).numpy()
    
    #Vjerojatnost oznake u vremenskom koraku
    y_probs = tf.concat([model.predict(data_point[3]) for data_point in dataset.as_numpy_iterator()], axis=0)
   
    #Povjerenje najizglednije oznake
    y_conf = tf.reduce_max(y_probs, axis=-1).numpy()
    
    #Predikcija oznake
    y_pred = tf.argmax(y_probs, axis=-1, output_type=tf.int32).numpy()
    
    #Stvarna oznaka
    y_true = tf.concat([data_point[4] for data_point in dataset.as_numpy_iterator()], axis=0).numpy()
    
    return y_lens, y_conf, y_pred, y_true
#%%
y_lens, y_conf, y_pred, y_true = extract_pred_data(model, "train")
#%%Izračun metrika
labels = []
preds = []
pred_conf = []

for num, pred, label, conf in zip(y_lens, y_pred, y_true, y_conf):
    labels.append(label[:num])
    preds.append(pred[:num])
    pred_conf.append(conf[:num])

#%%Izračun accuracy-a
TP = 0
total_samples = 0
for num, pred, label in zip(y_lens, preds, labels):
    #Ukupan broj vremenskih koraka (bez nadopune)
    total_samples += num
    #Stvarno pozitivni (bez nadopune)
    TP += (pred == label).sum()  
accuracy = (TP / total_samples) * 100

#%%Izračun F1@IoU
iou_threshold = [0.1, 0.25, 0.5]
f1_scores = [f1_at_IoU(labels, preds, bg_class=None, iou_threshold=threshold) for threshold in iou_threshold]

#%%Izračun mAP@IoU na 0.5
#Podatci za izračun mAP-a
true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, pred_conf = data_for_mAP(labels, preds, pred_conf)

#%%Zanima nas samo mAP, ostalo odbacujemo
mAP_score = []
for threshold in iou_threshold:
    *_, mAP = mAP_at_IoU(true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, pred_conf, iou_threshold=threshold)
    mAP_score.append(mAP)

#%%Provjera rezultata
train_path = r"D:\Phd_data\Video_feats_tfrecords\train_base\Fokus\train"
full_train_path = glob.glob(os.path.join(train_path, "*.tfrecord"))

train_dataset = build_test_pipeline(full_train_path, example2video(feature_dim=2048, training=True),
                                    batch_size=32, padded_batch=True, padded_shapes=([457, None], [457]))
#%%
model.evaluate(train_dataset)


#%%Biblioteka
import argparse
import os
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import read_labels
from phd_lib.models.ms_tcn_model import MaskConv1D, DilatedResidualModule, PredictionGeneration, Refeinment

#%%Dohvat ulaznog opažanja i stvarnih oznaka za to opažanje

#Parsiranje putanje modela
m_path = "CONV/train_base/Fokus"
#%%
MODEL_TYPE, FEAT_TYPE, KADAR = m_path.split("/")

#Riječnik do odgovarajućih značajki u npy formatu
feat_type_dict = {"feat_extraction": config.VIDEO_FEATS_31,
                  "transfer_learning": config.VIDEO_FEATS_32,
                  "train_base": config.VIDEO_FEATS_33}

#Odabir bazne putanje s obzirom na tip značajke i kadra na kojem je naučen model
base_feats_dir = os.path.join(feat_type_dict[FEAT_TYPE], KADAR)

#Generiranje pune putanje do odabranog uzorka AKO POSTOJI inače je to prazna lista
sample_path = glob.glob(os.path.join(base_feats_dir, "**", "161*.npy"), recursive=True)

#Ako odabrani uzorak NE postoji nasumično odaberi uzorak
if not sample_path:
    print("[INFO] odabrani broj uzorka se ne nalazi u postojećem skupu podataka - nasumično odabirem uzorak!")
    
    #Generiranje svih putanja za odgovarajuće značajke i kadar
    all_paths = glob.glob(os.path.join(base_feats_dir, "**", "*.npy"), recursive=True)
    
    #Izvlačenje putanje opažanja nasumično
    idx = random.randint(0, len(all_paths) - 1)
    sample_path = all_paths[idx]
     
sample_path = sample_path if isinstance(sample_path, str) else sample_path[0]

#Podizanje uzorka i oznake
sample_id = os.path.basename(sample_path).split(".")[0]
print(f"[INFO] odabran je uzorak: {sample_id}")
#Uzorak
sample = np.load(sample_path)
sample = np.expand_dims(sample, axis=0) #Dodavanje dimenzije - zahtjev modela
#Izvlačenje oznake odabranog opažanja
label_path = glob.glob(os.path.join(config.LABELS, "**", sample_id + ".txt" ), recursive=True)[0]   
labels = np.array(read_labels(label_path.split(".")[0]))

#%%Podizanje modela
print("[INFO] podizanje modela...")

#Dodajemo prvu i zadnju komponentu putanje modela
full_model_path = r"D:\Phd_data\Models\glava_modeli\CONV\train_base\Fokus\EX_1\epoch_40"

#Ako je model tipa "CONV" potrebno je kod podizanja registrirati custom komponente modela
#te pripremiti izlaz samo iz zadnjeg sloja
if MODEL_TYPE == "CONV":
    custom_objects = {"MaskConv1D": MaskConv1D, "DilatedResidualModule": DilatedResidualModule, 
                      "PredictionGeneration": PredictionGeneration, "Refeinment": Refeinment}
    
    model = load_model(full_model_path, custom_objects=custom_objects)
    
    #Izlaz samo iz zadnjeg sloja
    outputs = [layer.output for layer in model.layers]
    model = tf.keras.Model(inputs=model.input, outputs=outputs[-1])
else:
    model = load_model(full_model_path)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

#%%Generiranje predikcija iz modela
print("[INFO] generiranje predikcija iz modela...")
pred_probs = model.predict(sample)
preds = np.squeeze(np.argmax(pred_probs, axis=-1))
#%%Zapisivanje oznake i predikcije na svaku sličicu iz uzorka
frame_paths = glob.glob(os.path.join(config.INPUT_DATASET, "HE", "**", sample_id, "*.jpeg"), recursive=True)

#Lokacija za spremanje obilježenog opažanja (video zapisa)
output_dir = os.path.join(config.ANNOTATED_VIDEOS, FEAT_TYPE, KADAR, MODEL_TYPE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, f"{sample_id}.avi")

#Inicijalizacija pisača, izbor codec-a
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output_file, fourcc, 5,
			(320, 192), True)

#%%Petlja po sličici, predikciji i stvarnoj oznaci sličice
for path, pred, label in zip(frame_paths, preds, labels):
    #Otvaranje sličice
    frame = cv2.imread(path)
    
    frame = frame.copy()
    
    #Pridodavanje teksta predikcije i oznake
    text_pred = f"Preds: {config.ACT_KLASE[pred]}"
    text_label= f"Label: {config.ACT_KLASE[label]}"
    
    #Ako su predikcija i oznaka iste zapiši tekst u zelenoj boji, inače u crvenoj
    if pred == label:
        cv2.rectangle(frame, (0,0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, text_pred, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
        cv2.putText(frame, text_label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    else:
        cv2.rectangle(frame, (0,0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, text_pred, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1) #CV koristi BGR shemu !!!
        cv2.putText(frame, text_label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1)
    #Zapiši sličicu
    writer.write(frame)
    
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
 
#Otpusti pointer prema video datoteci
writer.release()
print(f"[INFO] obilježeni video se nalazi u: {output_file}")

  



	