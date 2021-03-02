#%%Biblioteke
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

#%%Funkcija za promjenu veličine slike
def resize_img(image_path, img_size=(224, 224)):
    '''Čitanje i promjena veličine slike. Vraća uint8 sliku.'''
    #Čitanje slike
    img = cv2.imread(image_path)
    #Promjena iz BGR u RGB redoslijed kanala
    img = img[:,:,::-1]
    #Promjena rezolucije u 224x224, zahtjev ResNet50 ulaza
    img = cv2.resize(img, img_size)
    
    return img

#%%Funkcija za pripremu slike obradom ResNet50 mrežom
def preprocess_img_for_ResNet50(image_path):
    '''Čitanje i priprema slike za daljnju obradu ResNet50 modelom. 
    Vraća float32 ndarray.'''
    #Čitanje slike
    img = cv2.imread(image_path)
    #Promjena iz BGR u RGB redoslijed kanala
    img = img[:,:,::-1]
    #Promjena rezolucije u 224x224, zahtjev ResNet50 ulaza
    img = cv2.resize(img, (224, 224))
    #ResNet50 predprocesuiranje slike
    img_array = preprocess_input(img)
    
    return img_array

