import os
import argparse
#Skuplja sve logove koje generira tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#Definiranje backend tako da se slike spremaju u .png fajl
mpl.use("Agg")
import seaborn as sns
from phd_lib import config
from phd_lib.data_pipeline.pipe_builders import build_test_pipeline
from phd_lib.models.ms_tcn_model import MaskConv1D, DilatedResidualModule, PredictionGeneration, Refeinment, SegmentationLoss 
from phd_lib.data_pipeline.tfrecord_helpers import example2video
from phd_lib.metrics import f1_at_IoU
import tensorflow as tf
from tensorflow.keras.models import load_model

#%%Parsiranje ulaznih argumenata
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True,
	help="putanja do naučenog modela sa strukturom: 'model_type/feat_type/kadar'")

ap.add_argument("-ds", "--data-split", type=str, default="test",
	help="za koju podjelu podataka da izračunam metriku: 'train', 'val' ili 'test'")

args = vars(ap.parse_args())

#%%Konfiguracija skripte
MODEL_TYPE, FEAT_TYPE, KADAR = args["model"].split("/")

#Putanje za odabir najboljeg modela za specifikaciju "MODEL_TYPE/FEAT_TYPE/KADAR"
MODEL_PATH = os.path.join(config.MODELS, "glava_modeli", args["model"], "model") #direktorij "model" sadrži najbolji model

#Putanja do direktorija za pohranu podataka
METRICS_DIR = os.path.join(config.VIS_METRIC, "_".join([MODEL_TYPE, FEAT_TYPE, KADAR]))

#Hiperparametri za testiranje
BATCH_SIZE = 16 if KADAR == "concat" else 32
FEATURE_DIM = 4096 if KADAR == "concat" else 2048

#%%Podizanje modela
print("[INFO] podizanje modela...")
#Ako je model tipa "CONV" potrebno je kod podizanja registrirati custom komponente modela
#te pripremiti izlaz samo iz zadnjeg sloja
if MODEL_TYPE == "CONV":
    custom_objects = {"MaskConv1D": MaskConv1D, "DilatedResidualModule": DilatedResidualModule, 
                      "PredictionGeneration": PredictionGeneration, "Refeinment": Refeinment, "SegmentationLoss": SegmentationLoss}
    
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    
    #Izlaz samo iz zadnjeg sloja
    outputs = [layer.output for layer in model.layers]
    model = tf.keras.Model(inputs=model.input, outputs=outputs[-1])
else:
    model = load_model(MODEL_PATH)

#%%Definiranje pipeline-a
print("[INFO] priprema ulaznog pipeline-a...")
#Pomoćna funkcija za definiranje putanja do značajki odgovarajućeg tipa, kadra i podjele podataka
def get_feature_files(feat_type, kadar, data_split="test"):
    feat_type_dict = {"feat_extraction": config.TFR_VIDEO_FEATS_31,
                  "transfer_learning": config.TFR_VIDEO_FEATS_32,
                  "train_base": config.TFR_VIDEO_FEATS_33}
    
    ulazni_dir = feat_type_dict[feat_type]
    filenames = glob.glob(os.path.join(ulazni_dir, 
                                       kadar, 
                                       data_split, 
                                       "*.tfrecord"), recursive=True)
    return filenames

#Lokacija ulaznih podataka
feat_path = get_feature_files(FEAT_TYPE, KADAR, data_split=args["data_split"])

#Definiranje dataset-a, broj vremenskih koraka nadopunjuje se do najvećeg broja koraka u cijelom skupu(457)
#kako bi kod izrade grafa imali optimiziranu implementaciju
#Struktura izlaza dataset-a: video_id, num_images, num_labels, image_seq, labels 
dataset = build_test_pipeline(feat_path, example2video(feature_dim=FEATURE_DIM, training=False),
                                  batch_size=BATCH_SIZE, padded_batch=True, 
                                  padded_shapes=([], [], [], [457, None], [457]), 
                                  padding_values=("", 0, 0, 0.,0))

#Izvlačenja Video_id-a
y_id = tf.concat([data_point[0] for data_point in dataset.as_numpy_iterator()], axis=0).numpy()

#Izvlačenja duljina nizova bez nadopuna
y_lens = tf.concat([data_point[1] for data_point in dataset.as_numpy_iterator()], axis=0).numpy()

#Vjerojatnost oznake u vremenskom koraku
y_probs = tf.concat([model.predict(data_point[3]) for data_point in dataset.as_numpy_iterator()], axis=0)

#Predikcija oznake
y_pred = tf.argmax(y_probs, axis=-1, output_type=tf.int32).numpy()

#Stvarna oznaka
y_true = tf.concat([data_point[4] for data_point in dataset.as_numpy_iterator()], axis=0).numpy()

#%%Izračun metrike
#Pretvori y_id u sample_nums
print("[INFO] izračun metrike...")
sample_nums = [sample.decode().split("_")[0] for sample in y_id]

#Izračun accuracy-a i F1@50 za svaki video zasebno
sample_accuracy = []
f1_at_50 = []
for num, pred, label in zip(y_lens, y_pred, y_true): 
    #Stvarno pozitivni (bez nadopune)
    TP = (pred == label)[:num].sum()
    sample_accuracy.append(round((TP/num)*100, 2))
    f1_at_50.append(round(f1_at_IoU(label[:num], pred[:num]), 2))

#Kreiranje DataFrame-a o broju uzorka, duljini uzorka i metrikama
metric_results = pd.DataFrame(data={"sample_num": sample_nums, "acc": sample_accuracy, 
                                    "f1_at_50": f1_at_50, "lens": y_lens})


#Pomoćna funkcija za izvlačenje podataka o najgorem ili najboljem uzorku s aspekta f1 metrike
def extract_sample_data(data, y_true, y_pred, best=True):  
    
    selector_func = np.argmax if best else np.argmin
    sample_info = data.loc[selector_func(data["f1_at_50"]), ["sample_num", "lens"]]
    sample_num, lens, idx = sample_info.sample_num, sample_info.lens, sample_info.name
    f1 = data.loc[idx ,"f1_at_50"]
    acc = data.loc[idx ,"acc"]
    preds = y_pred[idx, :lens]
    labels = y_true[idx, :lens]
    
    return sample_num, acc, f1, preds, labels

#Izvlačenje podataka za najgori i najbolji uzorak po f1 metrici
best_sample, best_acc, best_f1, best_preds, best_labels = extract_sample_data(metric_results, y_true, y_pred, best=True)
worst_sample, worst_acc, worst_f1, worst_preds, worst_labels = extract_sample_data(metric_results, y_true, y_pred, best=False)

#U istu strukturu
bests = np.concatenate((best_labels[np.newaxis], best_preds[np.newaxis]), axis=0)
worsts = np.concatenate((worst_labels[np.newaxis], worst_preds[np.newaxis]), axis=0)

#%%Spremanje podataka
print("[INFO] spremanje rezultata...")
if not os.path.exists(METRICS_DIR):
    os.makedirs(METRICS_DIR)
    
#spremi podatke o svim testni podatcima
metric_results.to_excel(os.path.join(METRICS_DIR, "all_results.xlsx"), float_format="%.2f", index=False)
#spremi predikcije i stvarne oznake za najbolje i najgore

bests_pd = pd.DataFrame(bests)
worsts_pd = pd.DataFrame(worsts)

bests_pd.to_excel(os.path.join(METRICS_DIR, f"{best_sample}.xlsx"), index=False)
worsts_pd.to_excel(os.path.join(METRICS_DIR, f"{worst_sample}.xlsx"), index=False)

#%%Vizualizacija rezultata
def seg_viz(x, sample_num, f1, acc):
    
    col_list = ["light yellow", "gunmetal","hot pink","baby blue","green blue", 
             "baby pink","blood red","magenta", "dusky blue","neon purple"]
    col_list_palette = sns.xkcd_palette(col_list)
    CustomCmap = mpl.colors.ListedColormap(col_list_palette)
    
    fig = plt.figure()
    fig.suptitle(t=f"Uzorak: {sample_num}, F1@50: {f1}%, Accuracy: {acc}%",
                 x=0.48, y=0.95, weight="semibold")
    
    plt.subplot(2,1,1)
    plt.imshow(x[0, np.newaxis], interpolation="nearest",cmap=CustomCmap,aspect=50)
    plt.ylabel("Oznaka")
    plt.yticks([])
    plt.xticks([])
    plt.axis("tight")
    
    plt.subplot(2,1,2)
    plt.imshow(x[1, np.newaxis], interpolation="nearest",cmap=CustomCmap,aspect=50)
    plt.ylabel("Predikcija")
    plt.yticks([])
    plt.xticks([])
    plt.axis("tight")
    
    plt.savefig(os.path.join(METRICS_DIR, f"{sample_num}.png"), dpi=640)
    
    
#%%Spremanje slika
print("[INFO] spremanje slika...")

seg_viz(bests, best_sample, best_f1, best_acc)
seg_viz(worsts, worst_sample, worst_f1, worst_acc)