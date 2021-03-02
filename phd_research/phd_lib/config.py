#%%Biblioteke
import os

#%%Konfiguracijska datoteka

#Nazivi klasa aktivnosti
ACT_KLASE = ["Pozadina", "Formiranje kuta", "Ucvrscivanje kopce LG",
             "Umetanje lamela", "Postavljanje poprecne stranice",
             "Postavljanje uzduzne stranice", "Odlaganje gotovog proizvoda",
             "Ucvrscivanje kopce DG", "Ucvrscivanje kopce DD",
             "Ucvrscivanje kopce LD",]

#Bazni direktorij za pohranu svih poddirektorija
BASE_PATH = os.path.sep.join(["D:", "Phd_data"])

#Lokacija video uzorka u obliku pojedinačnih sličica 
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID/SLIKA-####.jpeg" 
INPUT_DATASET = os.path.sep.join([BASE_PATH, "Uzorak_slika"])

#Lokacija oznaka svake sličice u videu
#Struktura poddirektorija "PODJELA_PODATAKA/VIDEO_ID.txt" => [video_id, oznake]
LABELS = os.path.sep.join([BASE_PATH, "Labels"])

#Lokacija za spremanje slika potpuno pripremljenih za obradu ResNet50 mrežom
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID/SLIKA-####.npy"
IMAGES_RESNET = os.path.sep.join([BASE_PATH, "ResNet50_ready_images"])

#Lokacija za spremanje slika čija je dimenzija podešena za obradu ResNet50 mrežom,
#ali je potrebno centiranje i i zamjena redoslijeda kanala
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID/SLIKA-####.npy"
IMAGES_RESIZED = os.path.sep.join([BASE_PATH, "Resized_images"])

#Lokacija za spremanje slika i oznaka za obradu ResNet50 mrežom u TFRecord formatu
#Slike su potpuno spremne za obradu ResNet50 mrežom
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID.tfrecord" => [image_id, image, label]
TFR_IMG_RESNET = os.path.sep.join([BASE_PATH, "Img_ResNet50_tfrecords"])

#Lokacija za spremanje slika i oznaka za obradu ResNet50 mrežom u TFRecord formatu
#Slike nisu potpuno spremne ResNet50 mrežom
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID.tfrecord" => [image_id, image, label]
TFR_IMG_RESIZED = os.path.sep.join([BASE_PATH, "Img_Resized_tfrecords"])

#Lokacija za spremanje video značajki izračunatih ResNet mrežom u npy formatu
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/VIDEO_ID.npy" => [video_id, video_feats, labels]
#Ovisno o načinu pripreme bazne mreže razlikuju se tri lokacije
VIDEO_FEATS_31 = os.path.sep.join([BASE_PATH, "Video_feats", "feat_extraction"])
VIDEO_FEATS_32 = os.path.sep.join([BASE_PATH, "Video_feats", "transfer_learning"])
VIDEO_FEATS_33 = os.path.sep.join([BASE_PATH, "Video_feats", "train_base"])

#Lokacija za spremanje video značajki izračunatih ResNet mrežom u TFRecord formatu
#Struktura poddirektorija: "KADAR/PODJELA_PODATAKA/#_grupa.tfrecord" => [video_id, video_feats, labels]
#Ovisno o načinu pripreme bazne mreže razlikuju se tri lokacije
TFR_VIDEO_FEATS_31 = os.path.sep.join([BASE_PATH, "Video_feats_tfrecords", "feat_extraction"])
TFR_VIDEO_FEATS_32 = os.path.sep.join([BASE_PATH, "Video_feats_tfrecords", "transfer_learning"])
TFR_VIDEO_FEATS_33 = os.path.sep.join([BASE_PATH, "Video_feats_tfrecords", "train_base"])

#Lokacija za spremanje naučenih modela
MODELS = os.path.sep.join([BASE_PATH, "Models"])

#Lokacija za spremanje rezultata metrike
RESULTS_METRIC = os.path.sep.join([BASE_PATH, "Metric_results"])

#Lokacija za spremanje anotiranih videa
ANNOTATED_VIDEOS = os.path.sep.join([BASE_PATH, "Annotated_videos"])

#Lokacija za spremanje vizualizacije metrike
VIS_METRIC = os.path.sep.join([BASE_PATH, "Metric_visualisation"])


