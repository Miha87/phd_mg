'''REFERENCA
F1@IoU definiran u:
    Lea, Colin, et al. "Temporal convolutional networks for action segmentation and detection." 
    proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

Dodatan izvor za implementaciju F1@IoU:
    https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py
'''
#%%Biblioteke
import numpy as np

#%%Helper privatne funkcije za izračun F1- za mapiranje segmentacijskih(po sličici) oznaka 
#u detekcijske oznake(po segmentu/aktivnosti)

#Segment/aktivnost u ovom kontekstu odnosi se na neprekinuti niz sličica
#tj. niz sličica sa istom oznakom

def _segment_labels(y):
    '''Vraća oznake segmenata/aktivnosti, prema njihovom redoslijedu pojave. 
    Uključene su i pozadinske aktivnosti. Ulaz je pojedinačno opažanje!!!'''
    
    #Pronađi razlike između susjednih vremenskih koraka - y[i+1] - y[i]
    razlike = np.diff(y)
    #U razlikama pronađi indekse različite od 0, to su završni indeksi trenutne
    #aktivnosti prije prelaska u novu aktivnost, zato im dodaj 1 da bi dobio početni indeks nove aktivnosti
    idxs = np.nonzero(razlike)[0] + 1
    #Dodaj 0 na početak - da dobiješ početni trenutak prve aktivnosti
    idxs = np.insert(idxs, 0, 0)
    #Petlja po indeksima, kako bi izvukao oznaku segmenta iz ulaznih oznaka
    labels = np.array([y[idx] for idx in idxs])
    
    return labels
    
def _segment_intervals(y):
    '''Vraća interval trajanja svake aktivnosti izražen kao [start, end>.
    Uključene su i pozadinske aktivnosti. Ulaz je pojedinačno opažanje.'''
    
    razlike = np.diff(y)
    idxs = np.nonzero(razlike)[0] + 1
    #Ovdje moramo dodati i 0, te duljinu ulaznog niza kako bi imali interval
    #izražen kao [start, end>
    idxs = np.insert(idxs, 0, 0)
    idxs = np.insert(idxs, len(idxs), len(y))
    #Petlja po duljini idxs
    intervals = np.array([(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)])
    
    return intervals

#%%Segmentacijska metrika
def f1_at_IoU(Y_true, Y_pred, bg_class=None, iou_threshold=.5):
    '''Izračun segmentacijske F1 metrike uz definiran minimalan prag 
    preklapanja između stvarne oznaka i predikcije. Kao ulaz može primiti i pojedinačno
    opažanje. Iz izračuna moguće isključiti pozadinsku klasu kroz definiranje 
    bg_class parametra.
    
    Ova metrika penalizira prekomjernu segmentaciju, ali ne penalizira male 
    vremenske pomake između stvarnih oznaka i predikcije kako bi kompenzirala
    varijabilnost u označavanju. Rezultat ovisi o broju aktivnosti ali ne i o
    duljini trajanja aktivnosti. Slična je mAP@IoU ali ne treba povjerenje
    za svaku predikciju.
    
        #1. Primjer 
            """python
            y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
            y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            
            score = f1_at_IoU(y_true, y_pred, iou_threshold=.5)
            """
            Primjer ilustrira da je ova mjera stroža od točnosti(accuracy),
            točnost za ovaj primjer iznosi 87,5, a F1 je 66,6. Ovdje je moguće
            uočiti i način na koji je penalizirana prekomjerna segmentiranost na primjeru
            uparivanja stvarnog segmenta oznake 0 sa predikcijom oznake 0.
            Razlog niske vrijednosti metrike posljedica je činjenice da 
            je model generirao dva segmenta s oznakom 0 i dva segmenta s oznakom 1.
        
        #2. Primjer
            """python
            y_pred = np.array([0, 1, 1, 1, 2, 2, 2, 2])
            y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
            
            score_1 = f1_at_IoU(y_true, y_pred, iou_threshold=.5)
            score_2 = f1_at_IoU(y_true, y_pred, iou_threshold=.6)
            """
            Ovaj primjer ilustira da pri pragu .5 metrika vraća 100%, što
            znači da metrika ne penalizira male vremenske pomake.
                
    Argumenti:
    ---------
    Y_true: list[ndarray]
        Lista nizova različitih duljina, dimenzije (broj_sličica,) 
        koji sadrže stvarne oznake za svaki vremenski korak opažanja.
    
    Y_pred: list[ndarray]
        Lista nizov različitih duljina, dimenzije (broj_sličica,) 
        koji sadrže predikciju oznake za svaki vremenski korak opažanja.
      
    bg_class: int ili None
        Numerička oznaka pozadinske klase, ako je želimo isključiti iz izračuna metrike.
    
    iou_threshold: float [0, 1]
        Minimalni IoU između duljine segmenta predikcije i stvarne oznake, kako bi 
        predikcija bila TP. Ako je preklapanje ispod overlap-a segment je računat kao FP. 
    
    Povratna vrijednost:
    -------------------
    Iznos F1 metrike u rasponu [0, 100], veće je bolje.
    '''
    #Provjera ulaznih parametara
    if iou_threshold < 0 or iou_threshold > 1.:
        raise ValueError("Preklapanje može biti samo u rasponu [0,1]")
    
    if not isinstance(Y_true, list):
        Y_true, Y_pred = [Y_true], [Y_pred]
    
    #Spremnici za broj stvarno pozitivnih, lažno pozitivnih i lažno negativnih
    TP = 0
    FP = 0
    FN = 0
    #Petlja po opažanjima
    for y_true, y_pred in zip(Y_true, Y_pred):
        #Izvlačenje oznaka segmenata i granica segmenata
        true_intervals = _segment_intervals(y_true)
        true_labels = _segment_labels(y_true)
        pred_intervals = _segment_intervals(y_pred)
        pred_labels = _segment_labels(y_pred)

        # Da li želimo ukloniti pozadinsku klasu iz izračuna metrike
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]
    
        #Spremanje ukupnog broja stvarnih segmenata i prepoznatih segmenata od strane modela 
        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        #Brojač iskorištenih stvarnih segmenata kod uparivanja sa predikcijom segmenata
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Petlja po predikciji segmenata i usporedba sa stvarnim segmentima preko IoU-a
            intersection = np.minimum(pred_intervals[j, 1], true_intervals[:, 1]) - np.maximum(pred_intervals[j, 0], true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(pred_intervals[j, 0], true_intervals[:, 0])
            #Osim što se koristi IoU, oznaka predikcije i stvarne oznake mora biti ista
            IoU = (intersection / union)*(pred_labels[j]==true_labels)
            #Odaberi indeks stvarnog segmenta sa kojim predikcija ima najveći preklop
            idx = IoU.argmax()
        
            # Ako je IoU veći od praga, te stvarni segment već nije uparen, računaj
            #predikciju segmenta kao TP i upari predikciju sa stvarnim segmentom, 
            #inače je uračunaj u FP
            if IoU[idx] >= iou_threshold and not true_used[idx]:
                TP += 1
                true_used[idx] = 1
            else:
                FP += 1
           
        #Lažno negativni su oni stvarni segmenti koji nisu upareni ni sa jednom predikcijom
        FN += (n_true - true_used.sum())
    
    #Izračunaj preciznost i odziv, te F1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision*recall) / (precision+recall)

    #U slučaju da je preciznost i odziv 0, postavi F1 na 0, a ne NaN
    F1 = np.nan_to_num(F1)
    
    return F1*100

#%%Helper privatne funkcije za izračun detekcijske metrike

def _iou_calculator(true_intervals, pred_intervals):
    '''Funkcija za izračun preklapanja između svih stvarnih segmenata
    i predikcije segmenata.
    
    Argumenti:
    ---------
    true_intervals: ndarray
        Niz dimenzije [m, 2] pri čemu je m broj stvarnih segmenata, a 2 odgovara
        [start, end> granicama segmenta.
     
    pred_intervals: ndarray
        Niz dimenzija [n, 2] pri čemu je n broj prepoznatih segmenat, a 2 odgovara
        [start, end> granicama segmenta.
     
    Povratna vrijednost:
    -------------------
    iou: ndarray
        Izračunati preklop između svih parova stvarnih i prepoznatih segmenta 
        dimenzije [m, n].
    '''
    if true_intervals.ndim != 2 or pred_intervals.ndim != 2:
        raise ValueError("Broj dimenzija ulaznih argumenta nije ispravan.")
    
    #Izvlačenje broja stvarnih i predikcije segmenata
    m, n = true_intervals.shape[0], pred_intervals.shape[0]
    #Spremnik za izračunate parove preklopa
    iou = np.zeros((m,n))
    
    #Petlja po stvarnim segmentima i usporedba sa svim predikcijama
    for i in range(m):
        union = np.maximum(true_intervals[i, 1], pred_intervals[:, 1]) - np.minimum(true_intervals[i, 0], pred_intervals[:, 0])
        inter = np.minimum(true_intervals[i, 1], pred_intervals[:, 1]) - np.maximum(true_intervals[i, 0], pred_intervals[:, 0])
        #Moguće je da presjek granica bude negativan, to nas ne zanima jer to znači da je presjek 0
        inter = inter.clip(0)
        iou[i] = inter/union
    
    return iou

def _average_precision_per_class(true_ids, pred_ids, true_labels, pred_labels, 
                   true_intervals, pred_intervals, label, pred_conf, iou_threshold):
    """
    Izračun prosječne preciznosti za odabranu klasu te akumuliranih preciznosti i 
    odziva za rangirane predikcije(detekcije) segmenata. 
    
    Da bi detekcija segmenta bila proglašena stvarno pozitivnom mora imati istu oznaku kao stvarni segment, te
    preklop s njim koji je veći od definiranog praga. Detekcije se rangiraju prema povjerenju (vjerojatnosti koju model
    pridodaje predikciji). Rangirane predikcije se zatim koriste za izračun akumuliranih preciznosti i odziva. 
    U slučaju detekcije preciznost je omjer stvarno pozitivnih segmenata te zbroja stvarno pozitivnih i lažno pozitivnih segmenata.
    Odziv je omjer stvarno pozitivnih segmenata i svih stvarnih segmenata. Završno, da bi izračunali prosječnu preciznost
    potrebno je akumulirane preciznosti za svaku detekciju množiti sa funkcijom relevantnosti detekcije (indikatorska funkcije koja je 1, ako
    je detekcija TP, inače 0), sve produkte sumirati i podjeliti sa brojem stvarnih segmenata.
    
    Argumenti:
    ---------
    true_ids : ndarray
        1D niz sa numeričkim oznakama pripadnosti stvarnih segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    pred_ids : ndarry
        1D niz sa numeričkim oznakama pripadnosti predikcije segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    true_labels : ndarray
        1D niz sa oznakama stvarnih segmenata.
        
    pred_labels : ndarray
        1D niz sa predikcijom oznaka prepoznatih segmenata.
        
    true_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici stvarnog segmenta.
        
    pred_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici predikcije segmenta.
        
    label : int
    Numerička oznaka klase za koju se računa metrika.
    
    pred_conf : ndarray
        1D niz sa iznosom povjerenja(vjerojatnosti) u predikciju segmenta.
        
    iou_threshold : float [0, 1]
        Prag IoU između predikcije i stvarnog segmenta tj. minimalni preklop za 
        uračunavanje predikcije kao TP-a.

    Povratna vrijednost:
    -------------------
    pr : ndarray
        Preciznost za izradu krivulje preciznosti i odziva. Akumulirana
        preciznost za rangirane segmente po povjerenju.
    
    rec: ndarray
        Odziv za izradu krivulje preciznosti i odziva. Akumulirani odziv
        za rangirane segmente po povjerenju.
    
    ap : float [0, 1]
        Prosječna preciznost odabrane klase.
    
    """
    #Pronađi koliko različitih opažanja ima u uzorku, (sortiraj ih)
    samples = np.unique(np.hstack([true_ids, pred_ids]))

    #Inicijalizacija spremnika za nerangirana povjerenja predikcija segmenta iz tražene klase
    unsorted_conf = []
    #Inicijalizacija spremnika za TP i FP predikcije segmenta iz tražene klase, povezan je s gornjim spremnikom
    unsorted_tp_fp = []
    
    #Izračunaj nazivnik odziva, iznimno za slučaju detekcije nazivnik je uvijek jednak
    #broju stvarno pozitivnih (TP) tj. broju stvarnih segmenata
    num_true_segments = float((true_labels == label).sum())

    # Ako predikcija ni jednog segmenta u svim videozapisima nema oznaku stvarne klase, 
    #odmah možemo vratiti metriku s vrijednostima 0
    if not any(pred_labels == label):
        return 0., 0., 0.

    #Petlja po opažanjima u uzorku
    for sample in samples:
        #Nađi indekse u listi oznaka svih opažanja gdje je trenutno opažanje jednako klasi za koju izračunavamo metriku
        true_idx = np.where((true_ids == sample)*(true_labels == label))[0]
        pred_idx = np.where((pred_ids == sample)*(pred_labels == label))[0]

        # Ako postoje predikcije segmenta sa traženom klasom u trenutnom opažanju
        if pred_idx.shape[0] > 0:
            # Sortiranje predikcija oznaka segmenata po povjerenju od najvećeg do najmanjeg zato(-)
            ind_s = np.argsort(-pred_conf[pred_idx])
            pred_idx = pred_idx[ind_s]
            conf = pred_conf[pred_idx]
            
            #Uparivanje predikcije segmenata sa stvarnim segmentima, ako nije uparen vrj. je 1
            ind_free = np.ones(len(pred_idx))

            # Ako postoje stvarni segmenti sa traženom klasom u trenutnom opažanju
            if true_idx.shape[0] > 0:
                #Izračunaj preklop između predikcije i stvarnih segmenata odabrane klase u trenutnom opažanju
                iou = _iou_calculator(true_intervals[true_idx], pred_intervals[pred_idx])
                #Petlja po dimenziji stvarnih segmenata (izlaz iz IoU kalkulatora), (m,n) dimenzija
                #jer ih je potrebno upariti sa predikcijom segmenta
                for m in range(iou.shape[0]):
                    #Koje predikcije segmenata još nisu uparene
                    ind = np.nonzero(ind_free)[0]
                    #Ako postoji još koja neuparena predikcija 
                    if len(ind) > 0:
                        #Indeks neuparene predikcija segmenta koja ima najveći preklop sa stvarnim segmentom m
                        ind_m = np.argmax(iou[m][ind])
                        #Vrijednost preklopa za gornji uvjet
                        val_m = iou[m][ind][ind_m]
                        #Provjeri da li je preklop veći od praga
                        if val_m > iou_threshold:
                            #Ako je, na mjesto uparene predikcije stavi 0
                            ind_free[ind[ind_m]] = 0
            
            #Pronađi indekse predikcija koje su uparene (0) to su TP, i one koje nisu uparene(1) to su FP
            ind_tps = np.where(ind_free == 0)[0]
            ind_fps = np.where(ind_free == 1)[0]

            #Označi sa 1 svaki TP, označi sa 2 svaki FP
            flag = np.hstack([np.ones(len(ind_tps)), 2 * np.ones(len(ind_fps))])
            #Sortiraj indekse uparenih i neuparenih predikcija
            ttIdx = np.argsort(np.hstack([ind_tps, ind_fps]))
            #Indeksi za povezivanjem predikcije sa pripadajućim povjerenjem
            idx_all = np.hstack([ind_tps, ind_fps])[ttIdx]
            #TP i FP po kronloškom redoslijedu
            flagall = flag[ttIdx]
            
            #Povjerenja svake predikcije segmenta dodano na postojeće
            unsorted_conf = np.hstack([unsorted_conf, conf[idx_all]])
            #TP i FP oznake obrađenih predikcija segmenata u trenutnom opažanju dodane na postojeće
            unsorted_tp_fp = np.hstack([unsorted_tp_fp, flagall])

    #Spoji sve dosad izračunate vrijednosti povjerenje, sa listom oznake da li je segment TP ili FP
    conf_and_tpfp=np.vstack([np.hstack(unsorted_conf), np.hstack(unsorted_tp_fp)])
    #Sortiraj indekse svih detekcije odabrane klase od najvećeg povjerenja do najmanjeg, zato (-)
    idx_s = np.argsort(-conf_and_tpfp[0])
    #Izračunaj broj akumuliranih stvarnih pozitivnih za rangirane predikcije po povjerenju
    TP = (conf_and_tpfp[1][idx_s] == 1).cumsum()
    #Izračunaj broj akumuliranih lažno pozitivnih za rangirane predikcije po povjerenju
    FP = (conf_and_tpfp[1][idx_s] == 2).cumsum()
    #Funkcija relevantnosti predikcije - indikatorska funkcija za izračun prosječne preciznosti, 
    #sa jedan označi ako je oznaka predikcije jednaka oznaci stvarne klase
    relevance = conf_and_tpfp[1][idx_s] == 1
    #Izračunaj akumulirani odziv za rangirane predikcije
    rec = TP / num_true_segments
    #Izračunaj akumuliranu preciznost za rangirane predikcije, tj. P@k, gdje je k broj segmenata koji su rangirani
    prec = TP / (TP + FP)
    # sum(P@k * relevance@k) / GTP (GTP = broj stvarnih segmenata)
    ap = np.sum(prec * relevance) / num_true_segments

    return rec, prec, ap

def data_for_mAP(Y_true, Y_pred, pred_conf):
    '''Priprema podataka za izračun mAP metrike.
    
    Argumenti:
    ---------
    Y_true : list[ndarray]
        Lista 1D nizova sa numeričkim oznakama stvarne klase svakog vremenskog koraka.
        Nizovi mogu biti različitih duljina.
        
    Y_pred : list[ndarry]
        Lista 1D nizova sa numeričkim oznakama predikcije klase svakog vremenskog koraka.
        Nizovi mogu biti različitih duljina.
        
    pred_conf : list[ndarray]
        Lista 1D nizova sa povjerenjem predikcije klase svakog vremenskog koraka.
        Nizovi mogu biti različitih duljina.
    
    Povratna vrijednost:
    -------------------
    true_ids : ndarray
        1D niz sa numeričkim oznakama pripadnosti stvarnih segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    pred_ids : ndarry
        1D niz sa numeričkim oznakama pripadnosti predikcije segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    true_labels : ndarray
        1D niz sa oznakama stvarnih segmenata.
        
    pred_labels : ndarray
        1D niz sa predikcijom oznaka prepoznatih segmenata.
        
    true_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici stvarnog segmenta.
        
    pred_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici predikcije segmenta.
        
    pred_conf : ndarray
        1D niz sa iznosom povjerenja(vjerojatnosti) u predikciju segmenta.
    '''
    #Inicijalizacija spremnika za stvarne segmente
    true_ids = []
    true_labels = []
    true_intervals = []
    
    #Petlja po opažanjima
    for i, y in enumerate(Y_true):
        #Izračun oznake cijelog segmenta
        true_labels += [_segment_labels(y)]
        #Izračun granica intervala [star,end>
        true_intervals += [_segment_intervals(y)]
        #Definiranje oznake uzorka koja će biti pridodana svim intervalima
        #i oznakama segmenta iz istog uzorka
        true_ids += [[i] * len(true_labels[-1])]
        
    #Naslagivanje svih oznaka    
    true_ids = np.hstack(true_ids)
    true_intervals = np.vstack(true_intervals)
    true_labels = np.hstack(true_labels)

    #Sličan postupak za predikciju
    pred_ids = []
    pred_labels = []
    pred_intervals = []
    conf = []
    for i, y in enumerate(Y_pred):
        pred_labels += [_segment_labels(y)]
        pred_intervals += [_segment_intervals(y)]
        pred_ids += [[i]*len(pred_labels[-1])]
        #Kod predikcije moramo u obzir uzeti i povjerenje u segment zbog rangiranja
        #iz razloga što svaki vremenski korak u segmentu ima vlastito povjerenje
        #odabirem najveće povjerenje kao reprezentaciju povjerenja segmenta
        conf += [pred_conf[i][inter[0]:inter[1]].max() for inter in pred_intervals[-1]]

    pred_ids = np.hstack(pred_ids)
    pred_intervals = np.vstack(pred_intervals)
    pred_labels = np.hstack(pred_labels)
    conf = np.hstack(conf)
    
    return true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, conf


#%%Detekcijska metrika

def mAP_at_IoU(true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, 
            pred_conf, iou_threshold=.5, bg_class=None):
    """
    Izračun detekcijske mAP(srednje uprosječene preciznosti) metrike uz definiran minimalan prag 
    preklapanja između stvarnog segmenta i predikcije. Vraća i prosječnu preciznost svake
    klase, te preciznost i odziv potreban za izračun krivulje preciznosti i odziva.
    
    Kao ulaz može primiti i pojedinačno opažanje. Iz izračuna moguće isključiti pozadinsku 
    klasu kroz definiranje bg_class parametra.
    
    Metrika je izračunata na slijedeći način:
        1) Prvo se računa prosječna preciznost (AP) za svaku klasu zasebno: 
                  1.a) Da bi detekcija segmenta bila proglašena stvarno pozitivnom mora imati istu oznaku 
                  kao stvarni segment te preklop s njim koji je veći od definiranog praga. 
                  1.b) Detekcije se rangiraju prema povjerenju (vjerojatnosti koju model pridodaje predikciji). 
                  1.c) Rangirane predikcije se zatim koriste za izračun akumuliranih preciznosti i odziva. 
                  U slučaju detekcije preciznost je omjer stvarno pozitivnih segmenata te zbroja stvarno pozitivnih i lažno pozitivnih segmenata.
                  Odziv je omjer stvarno pozitivnih segmenata i svih stvarnih segmenata. 
                  1.d) Završno, da bi izračunali prosječnu preciznost potrebno je akumulirane preciznosti za svaku detekciju množiti sa 
                  funkcijom relevantnosti detekcije (indikatorska funkcije koja je 1, ako je detekcija TP, inače 0) te sve produkte 
                  sumirati i podjeliti sa brojem stvarnih segmenata.
        2) Izračunati prosjek svih AP-ova.
                  
                  
        #Primjer 
            '''python
            Y_true = [np.array([0, 0, 0, 0, 1, 1, 1, 1]), 
                      np.array([0, 0, 1, 1, 2, 2, 2, 2])]
            
            Y_pred = [np.array([0, 1, 0, 0, 1, 1, 1, 1]), 
                      np.array([0, 1, 1, 1, 2, 2, 2, 2])]
            
            conf = [np.array([0.235, 0.157, 0.323, 0.255, 0.211, 0.821, 0.533, 0.152]),
                   np.array([0.135, 0.257, 0.423, 0.155, 0.311, 0.521, 0.733, 0.252])]
            
            true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, pred_conf = data_for_mAP(Y_true, Y_pred, conf)
            
            pr_all, rc_all, ap_all, mAP = mAP_at_IoU(true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, 
                               pred_conf, iou_threshold=.1)
            
            pr_all, rc_all, ap_all, mAP = mAP_at_IoU(true_ids, pred_ids, true_labels, pred_labels, true_intervals, pred_intervals, 
                               pred_conf, iou_threshold=.1, bg_class=0)
            '''
            Output #1:
            pr_all = [array([1.        , 0.5       , 0.66666667]),
                      array([1.        , 1.        , 0.66666667]),
                      array([1.])]
                
            rc_all = [array([0.5, 0.5, 1. ]), 
                          array([0.5, 1. , 1. ]), 
                          array([1.])]            
             
            ap_all = [0.833333, 1.0, 1.0]
 
            mAP = 0.9444443
                
            Output #2:
            pr_all = [array([1.        , 1.        , 0.66666667]),
                     array([1.])]
                
            rc_all = [array([0.5, 1. , 1. ]), 
                      array([1.])]            
             
            ap_all = [1.0, 1.0]
 
            mAP = 1.0
                 
    Argumenti:
    ---------
    true_ids : ndarray
        1D niz sa numeričkim oznakama pripadnosti stvarnih segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    pred_ids : ndarry
        1D niz sa numeričkim oznakama pripadnosti predikcije segmenata odgovarajućem uzorku.
        Npr. [0 0 1 1 1], znači da prva dva segmenta pripadaju uzorku 0, a slijedeća
        tri uzorku 1.
        
    true_labels : ndarray
        1D niz sa oznakama stvarnih segmenata.
        
    pred_labels : ndarray
        1D niz sa predikcijom oznaka prepoznatih segmenata.
        
    true_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici stvarnog segmenta.
        
    pred_intervals : ndarray
        2D niz oblika [broj_segmenata, 2] pri ćemu druga dimenzija 
        odgovara [start, end> sličici predikcije segmenta.
        
    pred_conf : ndarray
        1D niz sa iznosom povjerenja(vjerojatnosti) u predikciju segmenta.
        
    iou_threshold : float [0, 1]
        Prag IoU između predikcije i stvarnog segmenta tj. minimalni preklop za 
        uračunavanje predikcije kao TP-a. Preddefiniran vrijednost je .5.
        
    bg_class : int
        Numerička oznaka pozadinske klase, ako je želimo isključiti iz izračuna
        metrike. Preddefinirana vrijednost je None.

    Povratna vrijednost:
    -------------------
    pr_all : list[ndarray]
        Preciznost za izradu krivulje preciznosti i odziva.
    
    rec_all : list[ndarray]
        Odziv za izradu krivulje preciznosti i odziva.
    
    ap_all : list[float]
        Prosječna preciznost po svakoj klasi.
    
    mAP : float [0, 1]
        Srednja prosječna preciznost, izračunata kao prosjek svih AP-ova.
    """
    #Provjera ulaza
    if iou_threshold < 0 or iou_threshold > 1.:
        raise ValueError("Preklapanje može biti samo u rasponu [0,1]")
    
    #Izvlačenje oznaka različitih klasa
    labels = np.unique(true_labels)
    #Uklanjanje pozadinske klase iz izračuna
    if bg_class is not None and bg_class in labels:
        labels = np.delete(labels, labels == bg_class)
    
    #Inicijalizacija spremnika sa oznakama klase, preciznosti i odziva za crtanje krivulje 
    #preciznosti i odziva, te prosječne preciznosti po klasi
    pr_labels, pr_all, rec_all, ap_all = [], [], [], []
    #Petlja po različitim aktivnostima
    for label in labels:
        #Poziv funkciji koja izračunava metriku za pojedinu klasu - prosječnu preciznost po klasi, 
        rec, prec, ap = _average_precision_per_class(true_ids, pred_ids, true_labels, pred_labels, 
                                            true_intervals, pred_intervals, label, pred_conf, 
                                            iou_threshold)
        pr_labels += [label]
        pr_all += [prec]
        rec_all += [rec]
        ap_all += [ap]
    
    #Prosjek po svim prosječnim preciznostima pojedinih klasa aktivnosti
    mAP = np.mean(ap_all)

    return pr_all, rec_all, ap_all, mAP 
