# phd_mg
Repo sadrži materijal kreiran tijekom izrade doktorskog rada.

Tema rada bio je razvoj sustava za istovremeno prepoznavanje i vremensku segmentaciju  
ljudskih aktivnosti, primjenom dubokog strojnog učenja. Zainteresirani, mogu pretražiti   
literaturu o ovom problemu preko ključnih riječi **"action segmentation"** ili     
**"action detection"** iz video zapisa.

![25_gif](https://user-images.githubusercontent.com/34508474/109804982-2fa56480-7c23-11eb-86a3-8c17f60f4261.gif)
![196_gif](https://user-images.githubusercontent.com/34508474/109804991-3338eb80-7c23-11eb-9cb2-cb6c99a60b1d.gif)
![256_gif](https://user-images.githubusercontent.com/34508474/109805003-3633dc00-7c23-11eb-9815-57abe2f80911.gif)
![327_gif](https://user-images.githubusercontent.com/34508474/109805011-37fd9f80-7c23-11eb-8188-ab54e32b81dc.gif)

Izlaz sustava:  
![Bez naslova](https://user-images.githubusercontent.com/34508474/109805822-3385b680-7c24-11eb-8659-e461db392687.jpg)


Glavni elementi repo-a su:
* data_prep - priprema ulaznih podataka (ipynb)
* stat_analiza - statistička analiza uzorka iz phd-a (R)
* phd_research - sadrži razvijenu biblioteku **phd_lib** za duboko strojnog učenje   
s podatcima u obliku video zapisa i skripte u kojima je ta biblioteka primijenjena (Py).



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
Istražen je utjecaj snimaka iz *tri kadra snimanja*, uz *tri načina pripreme ulaznih značajki*   
i *tri arhitekture za finalnu segmentaciju i klasifikaciju*. Drugim riječima, napravljeni su  
eksperimenti s tri promijenjiva faktora pri čemu svaki faktor ima tri stanja.
