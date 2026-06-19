# underwater_debris_detection_yolo26
YOLOv26n underwater debris detection: evaluating  pool-to-sea domain shift and local data adaptation  on Adriatic sea-floor imagery.



# Detekcija podvodnog otpada pomoću YOLOv26n modela

Ovaj repozitorij sadrži popratne materijale za diplomski rad koji se bavi detekcijom podvodnog otpada i drugih objekata u podvodnom okruženju pomoću YOLOv26n modela.

Glavni cilj rada bio je ispitati kako se model treniran na javno dostupnim podvodnim skupovima podataka ponaša na novim snimkama iz stvarnog morskog okruženja te koliko lokalni podaci mogu poboljšati performanse modela na ciljnoj domeni.

## Opis projekta

U radu su korišteni javni skupovi podataka SeaClear i TrashCan, koji su spojeni u početni skup označen kao ST. Uz njih su korišteni i vlastiti skupovi podataka snimljeni u lokalnim uvjetima, uključujući Bašku, Muzzu i Biograd.

Eksperimenti su podijeljeni u dvije glavne skupine:

1. Progresivno dodavanje lokalnih podataka iz skupova Baška i Muzza u početni ST skup.
2. Progresivna prilagodba modela dodavanjem Biograd podataka, pri čemu je Biograd korišten kao ciljna domena za evaluaciju.

Model je treniran i evaluiran pomoću Ultralytics YOLO okvira.

## Klase

Svi skupovi podataka svedeni su na zajednički sustav od šest klasa:

```text
0: animal
1: trash_plastic
2: trash_other
3: nature
4: rov
5: unknown
```



## Objašnjenje foldera

### `configs/`

Folder `configs/data_yaml_examples/` sadrži primjere YOLO `data.yaml` konfiguracijskih datoteka. Datoteke ne sadrže stvarne putanje, nego generičke primjere s oznakom `PATH/TO/...`.

Prije pokretanja treninga ili evaluacije potrebno je zamijeniti te putanje stvarnim lokacijama skupova podataka.

### `scripts/`

Folder `scripts/` sadrži predloške skripti za trening i evaluaciju.

Datoteke su spremljene kao `.txt` radi lakšeg pregleda na GitHubu. Za stvarno pokretanje potrebno ih je preimenovati u `.py`, urediti putanje unutar skripti i zatim pokrenuti pomoću Pythona.

### `results/`

Folder `results/` sadrži rezultate eksperimenata.

* `results/figures/` sadrži grafove.
* `results/tables/` sadrži tablične rezultate tj. Excel tablice s metrikama.

### `examples/`

Folder `examples/` sadrži jednostavne primjere naredbi za pokretanje treninga i evaluacije.

## Eksperimentalni postupak

### Baseline

U početnom eksperimentu model je treniran samo na javnom ST skupu, koji se sastoji od SeaClear i TrashCan podataka. Taj model koristi se kao početna referenca za usporedbu s kasnijim eksperimentima.

### Runs1: dodavanje BM podataka

U prvom nizu eksperimenata postupno su dodavani podaci iz skupova Baška i Muzza. Dodavanje je provedeno u koracima od 10% do 100%.

Sekvenca treninga:

```text
T01: ST + 10% BM
T02: ST + 20% BM
T03: ST + 30% BM
T04: ST + 40% BM
T05: ST + 50% BM
T06: ST + 60% BM
T07: ST + 70% BM
T08: ST + 80% BM
T09: ST + 90% BM
T10: ST + 100% BM
```

Korišten je weight chaining, odnosno svaki sljedeći model inicijaliziran je najboljim težinama prethodnog modela.

### Runs2: prilagodba na Biograd

U drugom nizu eksperimenata korišten je najbolji model iz prvog niza eksperimenata te su mu postupno dodavane slike iz Biograd trening skupa.

Sekvenca treninga:

```text
S1: ST + 40% BM + Biograd step 1
S2: ST + 40% BM + Biograd step 2
S3: ST + 40% BM + Biograd step 3
S4: ST + 40% BM + Biograd step 4
S5: ST + 40% BM + Biograd step 5
S6: ST + 40% BM + Biograd step 6
```

Evaluacija je provedena na fiksnom Biograd evaluacijskom skupu.

## Evaluacijske metrike

Za evaluaciju modela korištene su standardne metrike za detekciju objekata:

* precision
* recall
* mAP50
* mAP50-95

Posebna pažnja posvećena je Biograd-only evaluaciji jer ona pokazuje koliko se model uspješno prilagodio ciljnoj domeni.

## Napomena o podacima

Zbog veličine i organizacije skupova podataka, stvarne slike, anotacije i trenirani modeli nisu uključeni u ovaj repozitorij.

U repozitoriju se nalaze:

* konfiguracijski primjeri
* skripte/predlošci za trening i evaluaciju
* tablični rezultati
* grafovi i vizualizacije rezultata

Korisnik koji želi pokrenuti skripte mora prilagoditi putanje prema vlastitoj lokalnoj strukturi podataka.

## Korišteni alati

Projekt je izrađen pomoću:

* Python
* Ultralytics YOLO
* PyTorch
* OpenCV
* Excel / CSV tablice za obradu rezultata

## Autor

Dario Perhat

Diplomski rad, Fakultet elektrotehnike i računarstva, Sveučilište u Zagrebu.
