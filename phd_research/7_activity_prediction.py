#PRIMJENA
#7_activity_prediction.py -m model_path -s sample_num
'''
Izrada obilježenog video zapisa za odabrani uzorak primjenom željenog modela.
  
Ulaz je putanja do modela, te broj uzorka za koji želimo vizualizaciju, pri čemu 
taj broj mora biti cjelobrojna vrijednost u rasponu [1, 620]. Ako ova vrijednost nije 
zadana ili je izvan raspona onda se nasumično odabire uzorak.
Iz putanje modela izvlače se informacije o ulaznim značajkama koje će biti prosljeđene
modelu za daljnju obradu. Struktura putanje je: 
    "model_type/feat_type/kadar" 
    Npr. ako je LSTM model naučen na značajkama izvučenima iz ResNet50 mreže
bez finog podešavanja(feat_extraction) i na kadru HE, to će biti ulazne značajke za evaluaciju modela.

Izlaz je obilježeni video zapis za odabrani uzorak. Pohrana slijedi strukturu: 
    "./Annotated_videos/feat_type/kadar/model_type/{sample_id}.avi"
'''
#%%Biblioteke
import argparse
import os 
#Skuplja sve logove koje generira tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import Counter
import random
import numpy as np
import cv2
import glob
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from phd_lib import config
from phd_lib.data_pipeline.tfrecord_helpers import read_labels
from phd_lib.models.ms_tcn_model import MaskConv1D, DilatedResidualModule, PredictionGeneration, Refeinment

#%%Parsiranje ulaznih argumenata
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True,
	help="Putanja do naučenog modela sa strukturom: 'model_type/feat_type/kadar'")

ap.add_argument("-s", "--sample", type=int, default=1234,
	help="Za koji uzorak (broj između 1-628) da vizualiziram predikciju, ako nije definirano nasumično odabirem opažanje")

ap.add_argument("-r", "--report", type=bool, default=False,
	help="Da li da generiram klasifikacijski izvještaj po aktivnostima, preddefinirana vrijednost je False")

args = vars(ap.parse_args())

#%%Dohvat ulaznog opažanja i stvarnih oznaka za to opažanje

#Parsiranje putanje modela
MODEL_TYPE, FEAT_TYPE, KADAR = args["model"].split("/")

#Riječnik do odgovarajućih značajki u npy formatu
feat_type_dict = {"feat_extraction": config.VIDEO_FEATS_31,
                  "transfer_learning": config.VIDEO_FEATS_32,
                  "train_base": config.VIDEO_FEATS_33}

#Odabir bazne putanje s obzirom na tip značajke i kadra na kojem je naučen model
base_feats_dir = os.path.join(feat_type_dict[FEAT_TYPE], KADAR)

#Generiranje pune putanje do odabranog uzorka AKO POSTOJI inače je to prazna lista
sample_path = glob.glob(os.path.join(base_feats_dir, "**", f"{args['sample']}_*.npy"), recursive=True)

#Ako odabrani uzorak NE postoji nasumično odaberi uzorak
if not sample_path:
    print("[INFO] odabrani broj uzorka se ne nalazi u postojećem skupu podataka - nasumično odabirem uzorak!")
    
    #Generiranje svih putanja za odgovarajuće značajke i kadar
    all_paths = glob.glob(os.path.join(base_feats_dir, "**", "*.npy"), recursive=True)
    
    #Izvlačenje putanje opažanja nasumično
    idx = random.randint(0, len(all_paths) - 1)
    sample_path = all_paths[idx]

#Prebaci tip podatka iz list-e u string ako je potrebno
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
full_model_path = os.path.sep.join([config.MODELS, "glava_modeli", args["model"], "model"])

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
    text_pred = f"Predikcija modela: {config.ACT_KLASE[pred]}"
    text_label= f"Stvarna oznaka: {config.ACT_KLASE[label]}"
    
    #Ako su predikcija i oznaka iste zapiši tekst u zelenoj boji, inače u crvenoj
    if pred == label:
        cv2.rectangle(frame, (0,0), (320, 40), (0, 0, 0), -1)
        cv2.putText(frame, text_label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
        cv2.putText(frame, text_pred, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    else:
        cv2.rectangle(frame, (0,0), (320, 40), (0, 0, 0), -1)
        cv2.putText(frame, text_label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1) #CV koristi BGR shemu !!!
        cv2.putText(frame, text_pred, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1)
    #Zapiši sličicu
    writer.write(frame)
    
    #Pokaži na ekran što se događa kod obilježavanja
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

 
#Otpusti pointer prema video datoteci
writer.release()
print(f"[INFO] obilježeni video se nalazi u: {output_file}")

#%%Vrijeme rada u uzorku
#Broj sličica po aktivnosti za stvarne oznake i predikcije
time_labs = Counter(labels)
time_preds = Counter(preds)

#Ispis rezultata
print()
print(f"{'Vrijeme trajanja aktivnosti':#^43}")
print(f"{'Aktivnost':32}", f"{'Labs':5}", f"{'Pred':5}")
print("-"*43)
for (key, lab), pred in zip(time_labs.items(), time_preds.values()):
    print(f"{config.ACT_KLASE[key]:32}", f"{round(lab * 0.2, 1):<5}",f"{round(pred * 0.2, 1):<5}")
print()
#%%Izvješće o rezultatima klasifikacije 
if args["report"]:
    print("[INFO] izvještaj o rezultatima klasifikacije uzorka")
    print(classification_report(labels, preds, target_names=config.ACT_KLASE))
