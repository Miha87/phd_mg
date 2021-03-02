#PRIMJENA
#6_evaluate_model.py -m model_path -ds data_split
'''
Izračun četiri tipa metrika za model na definiranoj podjeli podataka. 
Opcije za podjele podataka su: "train", "test" i "val". Zadana vrijednost je "test"

Metrike koje će biti izračunate su:
    a)Točnost po pojedinačnoj sličici (eng. Framewise accuracy)
    b)Srednje apsolutno odstupanja (eng. MAD) vremena trajanja aktivnosti - uprosječeno po vrsti aktivnosti i broju opažanja
    b)Segmentacijski F1 rezultat uz definiran prag preklapanja (10, 25, 50) (eng. F1@IoU)
    c)Srednja prosječna preciznost uz definiran prag preklapanja  (10, 25, 50) (eng. mAP@IoU)
    
Ulaz je putanja do modela, te podjela podataka za koju želimo rezultate.
Iz putanje modela izvlače se informacije o ulaznim značajkama koje će biti prosljeđene
modelu za daljnju obradu. Struktura putanje je: 
    "model_type/feat_type/kadar" 
    Npr. ako je LSTM model naučen na značajkama izvučenima iz ResNet50 mreže
bez finog podešavanja(feat_extraction) i na kadru HE, to će biti ulazne značajke za evaluaciju modela.

Izlaz je izračunata metrika u txt formatu pohranjena u odredišnoj lokaciji koja će biti ispisana
po izračunu metrike. Pohrana slijedi strukturu: 
    "./Metric_results/feat_type/kadar/model_type/{data_split}_results.csv"
'''
#%%Biblioteke
import argparse
import os
#Skuplja sve logove koje generira tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from collections import Counter
from phd_lib import config
from phd_lib.data_pipeline.pipe_builders import build_test_pipeline
from phd_lib.models.ms_tcn_model import MaskConv1D, DilatedResidualModule, PredictionGeneration, Refeinment, SegmentationLoss 
from phd_lib.data_pipeline.tfrecord_helpers import example2video
from phd_lib.metrics import mAP_at_IoU, f1_at_IoU, data_for_mAP
import tensorflow as tf
from tensorflow.keras.models import load_model

#%%Parsiranje ulaznih argumenata
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True,
	help="putanja do naučenog modela sa strukturom: 'model_type/feat_type/kadar'")

ap.add_argument("-ds", "--data-split", type=str, default="test",
	help="za koju podjelu podataka da izračunam metriku: 'train', 'val' ili 'test'")

args = vars(ap.parse_args())

#%%Podizanje modela
print("[INFO] podizanje modela...")
#Parsiranje putanje modela
MODEL_TYPE, FEAT_TYPE, KADAR = args["model"].split("/")

#Dodajemo prvu i zadnju komponentu putanje modela
full_model_path = os.path.sep.join([config.MODELS, "glava_modeli", args["model"], "model"])

#Ako je model tipa "CONV" potrebno je kod podizanja registrirati custom komponente modela
#te pripremiti izlaz samo iz zadnjeg sloja
if MODEL_TYPE == "CONV":
    custom_objects = {"MaskConv1D": MaskConv1D, "DilatedResidualModule": DilatedResidualModule, 
                      "PredictionGeneration": PredictionGeneration, "Refeinment": Refeinment, "SegmentationLoss": SegmentationLoss}
    
    model = load_model(full_model_path, custom_objects=custom_objects)
    
    #Izlaz samo iz zadnjeg sloja
    outputs = [layer.output for layer in model.layers]
    model = tf.keras.Model(inputs=model.input, outputs=outputs[-1])
else:
    model = load_model(full_model_path)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
#%%Definiranje putanje ulaznih značajki ovisno o tipu značajki i kadru na kojem je model naučen
print("[INFO] priprema ulaznog pipeline-a...")

#Definiranje putanje do odgovarajućih značajki tfr formatu
feat_type_dict = {"feat_extraction": config.TFR_VIDEO_FEATS_31,
                  "transfer_learning": config.TFR_VIDEO_FEATS_32,
                  "train_base": config.TFR_VIDEO_FEATS_33}

#Odabir bazne putanje s obzirom na tip značajke i kadra na kojem je naučen model
base_feats_dir = os.path.join(feat_type_dict[FEAT_TYPE], KADAR)

#Provjera dimenzionalnosti ulaznih značajki ovisno o kadru
FEATURE_DIM = 4096 if KADAR == "concat" else 2048

#Definiranje veličine mini grupe ovisno o kadru
BATCH_SIZE = 16 if KADAR == "concat" else 32

#Pomoćna funkcija za izvlačenje svih komponenata potrebnih za evaluaciju modela
#predikcija modela i povjerenja u predikcije, stvarnih oznaka te duljina nizova bez nadopuna
def extract_pred_data(model, data_split):
    #Putanje do značajki za odgovarajuću podjelu podataka
    feat_path = glob.glob(os.path.sep.join([base_feats_dir, data_split, "*tfrecord"]))
    
    #Definiranje dataset-a, broj vremenskih koraka nadopunjuje se do najvećeg broja koraka u cijelom skupu(457)
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
    
#%%Generiranje predikcija iz modela
print("[INFO] generiranje predikcija iz modela...")
y_lens, y_conf, y_pred, y_true = extract_pred_data(model, args["data_split"])

#%%Izračun metrike
print("[INFO] izračun metrike...")
#Metrike kao ulaz očekuju liste ndarray-eva
labels = []
preds = []
pred_conf = []

for num, pred, label, conf in zip(y_lens, y_pred, y_true, y_conf):
    labels.append(label[:num])
    preds.append(pred[:num])
    pred_conf.append(conf[:num])

#Izračun MAD za vrijeme trajanja aktivnosti
#Broj opažanja
N = 0
MSO_total = 0 
for pred, label in zip(preds, labels):
    N += 1
    #Broj sličica po određenom tipu aktivnosti za oznake i predikciju
    time_labs = Counter(label)
    time_preds = Counter(pred)
    #Broj vremenskih koraka i suma apsolutnih odstupanja po pojedinoj aktivnosti u opažanju
    T = 0
    SO = 0
    for act_pred, act_label in zip(time_labs.values(), time_preds.values()):
        T += 1
        #Izračun vremene pojedine aktivnosti za oznake i predikciju
        labs_time = round(act_label * 0.2, 1)
        pred_time = round(act_pred * 0.2, 1)
        #Izračun sume odstupanja
        SO += abs(labs_time - pred_time)
    #Izračun srednjeg apsolutnog odstupanja po aktivnostima u jednom opažanju i dodavanje ukupnom
    MSO_total += SO/ T
    #Ponovna inicijalizacija za slijedeće opažanje
    T = 0
    S0 = 0

MAD = MSO_total / N

#Izračun accuracy-a
TP = 0
total_samples = 0
for num, pred, label in zip(y_lens, preds, labels): 
    #Ukupan broj vremenskih koraka (bez nadopune)
    total_samples += num
    #Stvarno pozitivni (bez nadopune)
    TP += (pred == label).sum()
      
accuracy = (TP / total_samples) * 100

#Izračun F1@IoU
iou_threshold = [0.1, 0.25, 0.5]
f1_scores = [f1_at_IoU(labels, preds, bg_class=None, iou_threshold=threshold) for threshold in iou_threshold]

#Izračun mAP@IoU
#Podatci za izračun mAP-a
true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, pred_conf = data_for_mAP(labels, preds, pred_conf)

#Zanima nas samo mAP, ostalo odbacujemo
mAP_scores = []
for threshold in iou_threshold:
    *_, mAP = mAP_at_IoU(true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, pred_conf, iou_threshold=threshold)
    mAP_scores.append(mAP * 100)
    
#%%Pohrana rezultata metrike
print("[INFO] logiranje rezultata...")

metrics_path = os.path.sep.join([config.RESULTS_METRIC, FEAT_TYPE, KADAR, MODEL_TYPE])
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

full_metrics_path = os.path.join(metrics_path, f"{args['data_split']}_results.txt")

with open(full_metrics_path, "w") as f:
    f.writelines(["######### REZULTATI ##########\n",
                  "\n",
                  "Frame-wise točnost:\n",
                  f"Accuracy: {accuracy}\n",
                  "\n",
                  "MAD vremena trajanja aktivnosti:\n",
                  f"MAD: {MAD}\n",
                  "\n"
                  "Segmentacijski F1 rezultat uz minimalan prag preklapanja od 15, 25 i 50%\n", 
                  f"F1@10: {f1_scores[0]}\n",
                  f"F1@25: {f1_scores[1]}\n",
                  f"F1@50: {f1_scores[2]}\n",
                  "\n",
                  "Srednja prosječna preciznost uz minimalan prag preklapanja od 15, 25 i 50%\n",
                  f"mAP@10: {mAP_scores[0]}\n",
                  f"mAP@25: {mAP_scores[1]}\n",
                  f"mAP@50: {mAP_scores[2]}\n",
                  "\n"])

#%%Prikaz rezultata
with open(full_metrics_path, "r") as f:
    for line in f:
        print(line.rstrip())
print()     
print("Rezultati su pohranjeni u datoteku: ", full_metrics_path)

#%%Standardni testovi
test=0
if test:
    #Testiranje rezultata izračuna metrike
    feat_path = glob.glob(os.path.sep.join([base_feats_dir, args["data_split"], "*tfrecord"]))
    dataset = build_test_pipeline(feat_path, example2video(feature_dim=FEATURE_DIM, training=True),
                                  batch_size=BATCH_SIZE, padded_batch=True, 
                                  padded_shapes=([457, None], [457]))
    print(model.evaluate(dataset))