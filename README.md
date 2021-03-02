# phd_mg
Repo sadrži materijal kreiran tijekom izrade doktorskog rada.

Tema rada bio je razvoj sustava za istovremeno prepoznavanje i vremensku segmentaciju  
ljudskih aktivnosti, primjenom dubokog strojnog učenja. Zainteresirani, mogu pretražiti   
literaturu o ovom problemu preko ključnih riječi **"action segmentation"** ili     
**"action detection"** iz video zapisa.

Problem koji je motivirao istraživanje je želja da se olakša posao vrednovanja   
ljudskog faktora u proizvodnim procesima. Za ovo se obično koriste metode iz domene   
studija vremena. Postojeće metode su neefikasne, u smislu da zahtjevaju znatan   
utrošak vremenskog resursa, te su opterećene subjektivnosti analitičara.

Predloženo rješenje je model iz domene dubokog strojnog učenja koji iz ulaznog video   
zapisa može raspoznati koje aktivnosti ljudi izvode i koliko pojedine aktivnosti traju.  
To je posao koji se obično izvodi manualno od strane analitičara studija vremena,   
što ograničava učestalost provedbe studija vremena i pristup informacijama u stvarnom vremenu.

Napravljeno je 27 modela čiji su ulazi video zapisi trajanja do 2 minute,   
koji sadrže niz aktivnosti iz realnog proizvodnog procesa.  

Zašto 27 modela?  
Istraženi je utjecaj snimaka iz *tri kadra snimanja*, uz *tri načina pripreme ulaznih značajki*   
i *tri arhitekture za finalnu segmentaciju i klasifikaciju*. Drugim riječima, napravljeni su  
eksperimenti s tri promijenjiva faktora pri čemu svaki faktor ima tri stanja.

Glavni elementi repo-a su:
* Priprema ulaznih podataka
* Statistička analiza uzorka
* phd_research - sadrži razvijenu biblioteku **phd_lib** za duboko strojnog učenje   
s podatcima u obliku video zapisa i skripte u kojima je ta biblioteka primijenjena.
* testing - testiranje raznih funkcionalnosti TF 2, saznanja o konverziji video zapisa   
primjenom OpenCV-a itd.
