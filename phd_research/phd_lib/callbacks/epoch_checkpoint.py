#REFERENCA
'''
IMPLEMENTACIJA:
  Adrian Rosebrock, "Pyimagesearch" In: https://www.pyimagesearch.com 
  
  Moja izmjena je način praćenja trenutne etape preko argumenta "epoch"
'''
#PRIMJER:
'''python
    cp = EpochCheckpoint(output_path, every, start_at_epoch)
    model.fit(..., callbacks=[cp])
    
'''   
from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    '''Klasa za pohranu modela svaku n-tu epohu. 
    Ako počinjemo ispočetka start_at_epoch je 0.'''
    
    def __init__(self, output_path, every=5, start_at_epoch=0):
        # konstruktor super klase
        super(Callback, self).__init__()
		#Spremanje lokacije za pohranu modela
        #frekvencije spremanja tj. koliko epoha mora proći između pohrane
        #te u slučaju da nastavljamo trening iz koje epohe krećemo
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at_epoch
    
    def on_epoch_end(self, epoch, logs={}):
        '''Poziva se na kraju svake epohe'''
		# da li je potrebno pohraniti model
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path,
                                  f"epoch_{self.int_epoch + 1}"])
            self.model.save(p, overwrite=True)

        # ažuriranje brojača epoha
        self.int_epoch += 1

