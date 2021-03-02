#
'''
Modul podrazumijeva da je prethodno instaliran software FFMPEG!!!
'''
#%%Biblioteke
import os
import subprocess

#%%Funkcija za izvlačenje video isječaka iz dugog video zapisa

def video_to_segment(path_to_video, output_dir, output_file, start_point, duration):
    '''Pretvara dugi video u kraći isječak definiranog trajanja bez re-kodiranja 
    - veoma brza transforamacija kroz kopiranje ulaznog stream-a. 
    video zapisu dodaje nekoliko crnih sličica ispred i iza, ali kod
    pretvorbe iz videa u sliku, te sličica ffmpeg odbacuje, tako da 
    to nije problem kod označavanja.
    
    Argumenti:
    ---------
    path_to_video: str
        Apsolutna putanja do video zapisa.
    
    output_dir: str 
        Apsolutna putanja do direktorija u kojem će biti pohranjen isječak.
    
    output_file: str 
        Naziv datoteke izlaznog isječka.
    
    start_point: str
        Format "minuta:sekunda" u kojoj minuti i sekundi kreće isječak (npr. "9:15").
    
    duration: str 
        Format "minuta:sekunda" koliko je ukupno trajanje isječka.
    
    Povratna vrijednost:
    -------------------
    None
    '''
    #os.devnull služi za prikupljanje svih printova i njegovo odbacivanje
    with open(os.devnull, 'w') as ffmpeg_log: 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ffmpeg_call = ["ffmpeg",            #poziv s cmd line-a za ffmpeg, ovdje moramo opcije dati kao "string", kod cmd ne moramo
                       "-ss", start_point,  #opcija -ss, definiramo početak isječka, tj. koliko da preskočimo ulaznih sličica, ovo je brza opcija
                       "-i", path_to_video, #opcija -i, token opcije (argument) je path_to_video 
                       "-t", duration,      #opcija -t, definira trajanje isječka
                       "-c:v", "copy",      #opcija -c:v, definira codec koji će biti primjenjen na video, ovdje je to samo kopiranje
                       f"{output_dir}/{output_file}"] #Izlazna datoteka 
        
        subprocess.call(ffmpeg_call, stdout=ffmpeg_log, stderr=ffmpeg_log) #radi poziv cmd naredbi, i skuplja sve printeve
        
#%%Funkcija za izvlačenje slika iz video zapisa

def video_to_image(path_to_video, output_dir, image_res, fps):
    '''Pretvara video u niz slika
    
    Argumenti:
    ---------
    path_to_video: str
        Apsolutna putanja do video zapisa.
    
    output_dir: str
        Apsolutna putanja do direktorija u kojem će biti pohranjeni video zapisi.
    
    image_res: str 
        Format "širina:visina" definira rezoluciju izlazne slike.
    
    fps: float 
        Broj kojim definiramo frekvenciju uzorkovanja iz video zapisa.
    
    Povratna vrijednost:
    -------------------
    None
    '''
    #os.devnull služi za prikupljanje svih printova i njegovo odbacivanje
    with open(os.devnull, 'w') as ffmpeg_log: 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ffmpeg_call = ["ffmpeg",            #poziv s cmd line-a za ffmpeg, ovdje moramo opcije dati kao "string", kod cmd ne moramo
                       "-i", path_to_video, #opcija -i, token opcije (argument) je path_to_video
                       "-vf", f"scale={image_res}", #opcija -vf definira da radimo na video kanalu(filteru) za rezoluciju slike
                       "-qscale:v", "1",     #opcija -qscale na video stream-u, token opcije kontrolira kvalitetu slike, niže je bolje 
                       "-qmin", "1",         #opcija -qmin token opcije kontrolira donju granicu kvalitet
                       "-qmax", "1",         #opcija -qmax token opcije kontrolira gornju granicu kvalitet
                       "-r", f"{fps}",      #opcija -r, definra broj sličica u sekundi koji se uzorkuje
                       f"{output_dir}/slika-%04d.jpeg"] #Izlazne datoteke 
        
        subprocess.call(ffmpeg_call, stdout=ffmpeg_log, stderr=ffmpeg_log) #radi poziv cmd naredbi, i skuplja sve printeve
        
#%%Funkcija za spajanje dva videa
        
def concat_two_videos(path_to_video_1, path_to_video_2, output_dir, output_file):
    '''Spaja dva video zapisa u jedan, na taj način da prvi zapis postaje
    početni dio, a drugi zapis završni dio novog video zapisa
    
    Argumenti:
    ---------
    path_to_video_1: str
        Apsolutna putanja do prvog video zapisa.
    
    path_to_video_2: str
        Apsolutna putanja do drugog video zapisa.
    
    output_dir: str 
        Apsolutna putanja do direktorija u kojem će biti pohranjeni spojeni video zapis.
    
    output_file: str
        Naziv datoteke spojenog video zapisa.
    
    Povratna vrijednost:
    -------------------
    None
    '''
    #os.devnull služi za prikupljanje svih printova i njegovo odbacivanje
    with open(os.devnull, 'w') as ffmpeg_log: 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ffmpeg_call = ["ffmpeg",            #poziv s cmd line-a za ffmpeg, ovdje moramo opcije dati kao "string", kod cmd ne moramo
                       "-i", path_to_video_1, #opcija -i, token opcije (argument) je path_to_video_1
                       "-i", path_to_video_2, #opcija -i, token opcije (argument) je path_to_video_2
                       #opcija -filter_complex, koristimo opciju concat(spajanje), #n=2 (dva videa), v=1(postoji video), a=0 (ne postoji audio)
                       "-filter_complex", "concat=n=2:v=1:a=0", 
                       "-crf", "1",     #opcija -crf (constant rate factor), definira kvalitetu videa, u odnosu na veličinu
                      f"{output_dir}/{output_file}"] #Izlazni direktorij i datoteka
        
        subprocess.call(ffmpeg_call, stdout=ffmpeg_log, stderr=ffmpeg_log) #radi poziv cmd naredbi, i skuplja sve printeve

#%%Funkcija za pretvorbu podskupa slika u video
def images_to_video(path_to_images, start_image, end_image, output_dir, output_file):
    '''Pretvara podskup slika iz direktorija u video, pri čemu je moguće definirati
    početnu i završnu sliku, jako brzo jer se radi kopiranje.
    Preporuka je da format video zapisa bude .mkv
    
    Argumenti:
    ---------
    path_to_images: str 
        Apsolutna putanja do slika.
    
    start_image: int 
        Oznaka početne slike.
    
    end_image: int 
        Oznaka završne slike.
    
    output_dir: str 
        Apsolutna putanja do direktorija u kojem će biti pohranjeni spojeni video zapis.
    
    output_file: str 
        Naziv datoteke izlaznog video zapisa.
    
    Povratna vrijednost:
    -------------------
    None
    '''
    #os.devnull služi za prikupljanje svih printova i njegovo odbacivanje
    with open(os.devnull, 'w') as ffmpeg_log: 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ffmpeg_call = ["ffmpeg",              #poziv s cmd line-a za ffmpeg, ovdje moramo opcije dati kao "string", kod cmd ne moramo
                       "-framerate", "5",    #kod korištenja image file demuxera preporučeno je koristi ovu opciju a ne fps 
                       "-start_number", f"{start_image}", #Prva sličica u video zapisu
                       "-i", f"{path_to_images}/slika-%04d.jpeg", #opcija -i, token opcije (argument) je path_to_images
                       "-vframes", f"{end_image - start_image + 1}", #definiramo koliko sličica imamo u videu aka trajanje 
                                                                     # +1 tako da bude uključena i zadnja sličica      
                       "-c:v", "copy",      #opcija -c:v, video codec radi samo kopiranje u ovom slučaju
                       f"{output_dir}/{output_file}"] #Izlazni direktorij i datoteka
        
        subprocess.call(ffmpeg_call, stdout=ffmpeg_log, stderr=ffmpeg_log) #radi poziv cmd naredbi, i skuplja sve printeve

#%%Funkcija za pretvorbu podskupa videa u gif
def video_to_gif(path_to_video, output_dir, output_file, image_res, fps):
    '''Pretvara video zapis u gif
    
    Argumenti:
    ---------
    path_to_video: str
        Apsolutna putanja do video zapisa.
     
    output_dir: str
        Apsolutna putanja do direktorija u kojem će biti pohranjeni gif.
    
    output_file: str 
        Naziv datoteke izlaznog gif-a (npr. "ime.gif")
    
    image_res: str 
        Format "širina:visina" definira rezoluciju izlazne slike.
    
    fps: float 
        Broj kojim definiramo frekvenciju uzorkovanja iz video zapisa.
    '''
    #os.devnull služi za prikupljanje svih printova i njegovo odbacivanje
    with open(os.devnull, 'w') as ffmpeg_log: 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ffmpeg_call = ["ffmpeg",             #poziv s cmd line-a za ffmpeg, ovdje moramo opcije dati kao "string", kod cmd ne moramo
                       "-i", f"{path_to_video}", #opcija -i, token opcije (argument) je path_to_images
                       "-vf", f"fps={fps}, scale={image_res}", #opcija -vf definira da radimo na video kanalu, za opcije rezolucije i fps-a
                       "-loop", "0",      #opcija -loop, koliko ponavljnja imamo, ako je "0" onda je to beskonačna petlja
                       f"{output_dir}/{output_file}"] #Izlazni direktorij i datoteka 
        
        subprocess.call(ffmpeg_call, stdout=ffmpeg_log, stderr=ffmpeg_log) #radi poziv cmd naredbi, i skuplja sve printeve
