---
title: "Analiza sentimenta recenzija filmova"
subtitle: "Projekt iz kolegija Primjena umjetne inteligencije"
author: "Luka Ivanic"
date: "2026-06-04"
lang: hr-HR
---

# Sazetak

U ovom projektu razvijen je sustav za binarnu klasifikaciju sentimenta filmskih
recenzija. Ulaz u sustav je tekst recenzije, a izlaz je oznaka `positive` ili
`negative`. Projekt odgovara temi 7 iz materijala kolegija: "Analiza sentimenta
recenzija (tekst klasifikacija)". Prema zadatku, naglasak je na TF-IDF prikazu
teksta, usporedbi najmanje deset klasifikatora, koristenju
`RandomizedSearchCV` postupka s unakrsnom validacijom, racunanju standardnih
metrika i izradi zavrsnog `VotingClassifier` ili `StackingClassifier`
ensemble modela.

Primarni skup podataka u eksperimentima je Kaggle IMDb 50K Movie Reviews, s
50 000 oznacenih recenzija. Dodatno je preuzet i Stanford IMDb Large Movie
Review Dataset jer je naveden u materijalima kolegija kao drugi izvor za istu
temu. Nije bilo rucnog oznacavanja podataka: svi modeli uce iz vec oznacenih
recenzija.

Najbolji klasicni model bio je `LinearSVC` nad TF-IDF znacajkama s tocnosti
`0.9150` i F1 mjerom `0.9152` na izdvojenom testnom skupu. Zavrsni ensemble
modeli su implementirani pomocu `VotingClassifier` i `StackingClassifier`.
Najbolji ensemble po tocnosti bio je hard `VotingClassifier` s tocnosti
`0.9107`, dok soft `VotingClassifier` daje i probabilisticke metrike, ukljucivo
log-loss. Kao istrazivacko prosirenje, istrenirani su i transformerski modeli.
Najbolji ukupni rezultat postigao je fino ugodeni `microsoft/deberta-v3-small`
s tocnosti `0.9564`, F1 mjerom `0.9566` i ROC-AUC vrijednosti `0.9895`.
Transformerski modeli nisu zamjena za obavezni TF-IDF dio, nego dodatna analiza
modernijeg pristupa istom problemu.

# 1. Uvod

Analiza sentimenta je zadatak automatskog prepoznavanja stava ili emocionalnog
tona u tekstu. U ovom projektu radi se o pojednostavljenoj, ali prakticnoj
varijanti problema: recenzija filma klasificira se kao pozitivna ili negativna.
Takav problem se pojavljuje u preporucnim sustavima, analizi korisnickih
komentara, pracenju reputacije proizvoda i automatskoj obradi velikih zbirki
tekstova.

Za razliku od numerickih ili tablicnih problema, tekst nije izravno pogodan za
klasicne algoritme strojnog ucenja. Potrebno ga je pretvoriti u vektor
znacajki. U zadatku je propisano koristenje TF-IDF prikaza. TF-IDF je
jednostavan, ali jak prikaz teksta: dokument se opisuje rijecima i n-gramima
koji se pojavljuju u njemu, pri cemu se smanjuje utjecaj rijeci koje se
pojavljuju u gotovo svim dokumentima.

Ovaj rad prvo obraduje klasicni zadani dio projekta:

- ucitavanje IMDb recenzija,
- pretvorbu teksta u TF-IDF znacajke,
- treniranje i usporedbu deset propisanih klasifikatora,
- racunanje metrika,
- RandomizedSearchCV eksperiment s unakrsnom validacijom,
- zavrsni Voting/Stacking ensemble.

Nakon toga prikazano je dodatno prosirenje s neuronskim modelima. To prosirenje
nije potrebno za osnovno ispunjenje zadatka, ali pomaze razumjeti koliko daleko
se moze doci s modernim unaprijed naucenim jezicnim modelima u usporedbi s
klasicnim TF-IDF modelima.

# 2. Definicija zadatka

Prema dokumentu `PUI - Osnovne Informacije o kolegiju.pdf`, tema 7 je:

> Analiza sentimenta recenzija (tekst klasifikacija): TF-IDF + klasifikatori;
> usporediti modele i sloziti ensemble (voting/stacking).

Zadatak dalje navodi deset klasifikatora koje treba koristiti:

- `MultinomialNB`
- `ComplementNB`
- `LogisticRegression`
- `LinearSVC`
- `SGDClassifier`
- `PassiveAggressiveClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `XGBoostClassifier`
- `LightGBMClassifier`

Obavezni su i:

- najmanje deset sklearn klasifikatora,
- `RandomizedSearchCV` s unakrsnom validacijom,
- metrike ACC, BACC, precision, recall, F1, ROC-AUC, PR-AUC, MCC, log-loss i
  matrica zabune,
- zavrsni `VotingClassifier` ili `StackingClassifier`.

U ovom projektu zadatak je interpretiran kao nadzirana binarna klasifikacija:

- ulaz: tekst jedne filmske recenzije,
- izlaz: negativan ili pozitivan sentiment.

# 3. Skup podataka

Materijali kolegija navode dva moguca izvora:

- Kaggle: IMDb 50K Movie Reviews,
- Stanford: IMDb Large Movie Review Dataset.

Oba skupa su preuzeta lokalno:

- `data/raw/kaggle_imdb_50k/IMDB Dataset.csv`,
- `data/raw/stanford_imdb/aclImdb`.

Eksperimenti u izvjestaju koriste Kaggle IMDb 50K CSV jer je u njemu cijeli
skup dostupan u jednoj tablici s tekstom recenzije i oznakom sentimenta. Takav
oblik je pogodan za ponovljive eksperimente, izradu jedinstvenog train/test
splita i usporedbu vise modela pod istim uvjetima. Stanford skup je zadrzan u
repozitoriju kao dodatni profesorom navedeni izvor i kao provjera da tema ima
standardni, javno oznaceni izvor podataka.

Skup Kaggle IMDb 50K sadrzi:

- 50 000 recenzija,
- dvije oznake sentimenta: `positive` i `negative`,
- uravnotezen broj pozitivnih i negativnih primjera.

Za glavne eksperimente koristen je stratificirani split:

- 40 000 recenzija za treniranje,
- 10 000 recenzija za testiranje,
- `random_state=42`,
- testni skup s 5 000 negativnih i 5 000 pozitivnih recenzija.

Nije provedeno rucno oznacavanje. To je vazno jer je jedan od kriterija pri
odabiru teme bio izbjegavanje projekata koji zahtijevaju manualno labeliranje
novih podataka.

# 4. Metrike

Prije opisa vektorizacije i modela vazno je definirati kako se usporeduju
rezultati. Svi modeli imaju isti testni skup, pa su metrike izracunate nad istih
10 000 primjera. Za svaki model racunate su sljedece vrijednosti:

- accuracy: udio tocnih predikcija,
- balanced accuracy: prosjek odziva po klasama,
- precision: koliko pozitivnih predikcija je stvarno pozitivno,
- recall: koliko stvarno pozitivnih primjera je pronadeno,
- F1: harmonijska sredina precision i recall vrijednosti,
- ROC-AUC: sposobnost rangiranja pozitivnih primjera iznad negativnih,
- PR-AUC: povrsina ispod precision-recall krivulje,
- MCC: Matthews correlation coefficient, korelacijska mjera pogodna za binarnu
  klasifikaciju,
- log-loss: kazna za probabilisticke predikcije,
- matrica zabune.

Kod nekih modela log-loss nije prikazan. To nije propust u racunanju, nego
posljedica prirode modela. `LinearSVC`, `SGDClassifier` s hinge loss funkcijom
i `PassiveAggressiveClassifier` u ovoj konfiguraciji ne daju kalibrirane
vjerojatnosti klasa. Za njih se mogu racunati tvrde predikcije i rangirajuce
vrijednosti (`decision_function`) za ROC-AUC/PR-AUC, ali log-loss trazi
vjerojatnosnu distribuciju po klasama. Umjetno racunanje log-loss metrike iz
samo tvrdih predikcija dovelo bi do beskonacnih ili neinformativnih kazni, pa
je u rezultatima oznaceno da log-loss nije dostupan za te modele. Za modele
koji daju `predict_proba`, log-loss je izracunat.

![Slika 1. Metrike za binarnu klasifikaciju sentimenta: matrica zabune definira
TN, FP, FN i TP, iz kojih se izvode accuracy, balanced accuracy, precision,
recall, F1 i MCC; ROC-AUC/PR-AUC koriste rangiranje, a log-loss trazi
probabilisticki izlaz.](figures/generated/evaluation_metrics.png){width=92%}

# 5. Pretprocesiranje i TF-IDF prikaz

Klasicni modeli ne rade izravno nad nizovima znakova. Tekst je zato pretvoren u
rijetke numericke vektore pomocu `TfidfVectorizer` iz biblioteke scikit-learn.
U projektu se koristi isti osnovni TF-IDF postupak za sve klasicne modele, kako
bi usporedba modela bila sto postenija.

![Slika 2. Tok podataka u projektu: sirove IMDb recenzije se dijele na train i
test skup, iz trening skupa se uci TF-IDF vokabular, tekst se pretvara u
rijetku matricu znacajki, a klasifikator predvida pozitivan ili negativan
sentiment.](figures/generated/pipeline_labeled.png){width=95%}

Glavne postavke vektorizatora su:

- pretvorba teksta u mala slova,
- `strip_accents="unicode"`,
- `ngram_range=(1, 2)`, odnosno unigrami i bigrami,
- `min_df=2`, uklanjanje izraza koji se pojavljuju prerijetko,
- `max_df=0.95`, uklanjanje izraza koji se pojavljuju u gotovo svim dokumentima,
- `max_features=50000`,
- `sublinear_tf=True`.

TF-IDF ima dvije komponente. TF dio mjeri koliko se cesto izraz pojavljuje u
dokumentu. IDF dio smanjuje tezinu izraza koji se pojavljuju u mnogo dokumenata.
Ako se neka rijec pojavljuje gotovo svugdje, ona vjerojatno nije jako korisna
za razlikovanje pozitivnih i negativnih recenzija. S druge strane, rijeci ili
fraze koje su ceste u jednoj recenziji, a rijetke u cijelom skupu, dobivaju
vecu tezinu.

Vazno je da se TF-IDF vokabular uci samo na trening skupu. Testni skup se zatim
transformira pomocu vec naucenog vokabulara. Time se izbjegava curenje
informacija iz testnog skupa u postupak treniranja.

![Slika 3. Intuitivni prikaz TF-IDF postupka: vokabular se uci samo na trening
recenzijama, rijeci i bigrami postaju znacajke, TF mjeri lokalnu vaznost, IDF
smanjuje utjecaj preopcih izraza, a rezultat je rijedak vektor za
klasifikator. Slika naglasava i da TF-IDF nije BPE tokenizacija, nego fiksni
vokabular rijeci i n-grama.](figures/generated/tfidf_intuitive_generated.png){width=95%}

# 6. Modeli

![Slika 4. Pregled deset propisanih klasifikatora: Bayesovi modeli uce
vjerojatnosti rijeci, linearni modeli uce tezine i margine, tree ensemble modeli
glasaju preko stabala, a boosting modeli dodaju stabla koja ispravljaju
prethodne pogreske.](figures/generated/ten_classifiers_overview_generated.png){width=95%}

Ovo poglavlje je strukturirano kao usporedni "model atlas": prvo se zadrzava
isti TF-IDF prikaz, a zatim se mijenja algoritam ucenja. Takva struktura slijedi
uobicajen nacin prikaza usporednih tekstualnih klasifikacijskih eksperimenata:
opis zadatka i vektorizacije, zatim modeli po skupinama, pa zajednicke metrike i
rezultati. Kao korisni vanjski orijentiri posluzili su scikit-learn primjer
klasifikacije tekstnih dokumenata rijetkim TF-IDF znacajkama i noviji IMDb
usporedni radovi koji odvajaju klasicne TF-IDF modele od neuronskih ili
transformerskih prosirenja. Zbog toga su kartice modela ukljucene ovdje, u
glavnom tekstu, umjesto da budu zakopane u dodatku.

## 6.1 Naivni Bayesovi modeli

### 6.1.1 MultinomialNB

![Slika 5. `MultinomialNB`: model uci vjerojatnosti rijeci po klasi, koristi
alpha smoothing i predvida sentiment zbrajanjem log-vjerojatnosti za rijeci iz
TF-IDF prikaza.](figures/generated/classifier_cards/multinomial_nb.png){width=95%}

`MultinomialNB` i `ComplementNB` su varijante naivnog Bayesovog klasifikatora
koje su cesto pogodne za tekst. Pretpostavljaju uvjetnu nezavisnost znacajki
unutar klase. Ta pretpostavka nije doslovno tocna za prirodni jezik, ali modeli
su brzi, stabilni i cesto vrlo jaki kao pocetni tekstualni baseline.

`MultinomialNB` procjenjuje vjerojatnost znacajki po klasi i zatim za novu
recenziju zbraja log-vjerojatnosti rijeci ili bigrama koje se pojavljuju u
TF-IDF vektoru. Osnovni model postigao je tocnost `0.8841` i F1 `0.8846`.
Rezultat je vrlo dobar s obzirom na jednostavnost modela. Nedostatak je to sto
model ne uci odnose izmedu znacajki: rijeci i fraze se tretiraju kao uvjetno
nezavisni signali.

Tuned `MultinomialNB` koristi `RandomizedSearchCV` s 10 iteracija i 3 folda.
Njegova testna tocnost je `0.8854`, sto je malo poboljsanje, ali potvrduje da
TF-IDF parametri i `alpha` smoothing utjecu na rezultat. Ipak, vece poboljsanje
u projektu dolazi od prelaska na diskriminativne linearne modele.

### 6.1.2 ComplementNB

![Slika 6. `ComplementNB`: model procjenjuje znacajke iz komplementa klase, sto
je cesto korisno kod neuravnotezenih tekstualnih skupova; na uravnotezenom IMDb
splitu dao je isti rezultat kao `MultinomialNB`.](figures/generated/classifier_cards/complement_nb.png){width=95%}

`ComplementNB` koristi informacije iz komplementa klase i cesto je robusniji kod
neuravnotezenih tekstualnih skupova. U ovom projektu je skup gotovo savrseno
uravnotezen, pa su rezultati prakticki identicni kao kod osnovnog
`MultinomialNB`: tocnost `0.8841` i F1 `0.8846`.

Ovaj rezultat ne znaci da je `ComplementNB` los model, nego da dataset ne
naglasava njegovu glavnu prednost. Oba Bayesova modela zato sluze kao brzi,
interpretabilni baseline: hvataju dosta sentiment signala iz samih rijeci, ali
ne uce diskriminativnu granicu jednako dobro kao linearni modeli.

## 6.2 Linearni modeli

`LogisticRegression`, `LinearSVC`, `SGDClassifier` i
`PassiveAggressiveClassifier` su linearni modeli. Kod TF-IDF znacajki, broj
dimenzija je velik, ali je matrica vrlo rijetka. Linearni modeli su zato
prirodan izbor jer se mogu ucinkovito trenirati nad velikim rijetkim vektorima.

### 6.2.1 LogisticRegression

![Slika 7. `LogisticRegression`: diskriminativni linearni model uci tezine
TF-IDF znacajki, pretvara linearni score u vjerojatnost sigmoidnom funkcijom i
moze prirodno dati log-loss.](figures/generated/classifier_cards/logistic_regression.png){width=95%}

`LogisticRegression` uci linearnu granicu odluke i daje probabilisticke izlaze.
U projektu je postigla tocnost `0.9095` i F1 `0.9101`, sto je velik skok u
odnosu na Bayesove modele. Razlog je diskriminativni cilj: model izravno uci
tezine znacajki koje razdvajaju pozitivnu i negativnu klasu.

Najvaznija prakticna prednost je `predict_proba`. Zato logisticka regresija
daje log-loss, ROC-AUC i PR-AUC na standardan nacin. Ako bi sustav trebao
vracati pouzdanu procjenu sigurnosti, logisticka regresija bi bila jedan od
najboljih jednostavnih kandidata.

### 6.2.2 LinearSVC

![Slika 8. `LinearSVC`: linearni SVM trazi granicu odluke s velikom marginom u
rijetkom TF-IDF prostoru, optimira hinge loss i daje `decision_function` umjesto
kalibriranih vjerojatnosti.](figures/generated/classifier_cards/linear_svc.png){width=95%}

`LinearSVC` optimira SVM cilj s marginom. To je najbolji obavezni klasicni model
u projektu: tocnost `0.9150`, F1 `0.9152`, ROC-AUC `0.9720` i MCC `0.8300`.
SVM pristup je posebno dobar za visokodimenzionalne rijetke prostore jer uci
linearnu granicu koja ne samo da razdvaja klase, nego ih razdvaja s marginom.

Nedostatak konfiguracije koristenje u projektu je izostanak kalibriranih
vjerojatnosti. `LinearSVC` daje `decision_function`, sto je dovoljno za ROC-AUC
i PR-AUC, ali ne i za standardni log-loss. Da je cilj produkcijski sustav s
pouzdanim vjerojatnostima, mogla bi se dodati kalibracija, ali za propisanu
usporedbu ovaj rezultat je najbolji klasicni baseline.

### 6.2.3 SGDClassifier

![Slika 9. `SGDClassifier`: linearni model treniran stohastickim gradijentnim
postupkom; u projektu koristi hinge loss pa se ponasa slicno linearnom SVM-u,
ali s iterativnim stochastic update koracima.](figures/generated/classifier_cards/sgd_classifier.png){width=95%}

`SGDClassifier` omogucuje treniranje linearnog modela stohastickim gradijentnim
postupkom. U ovom eksperimentu koristen je hinge loss, sto ga cini slicnim
linearnom SVM-u. Postigao je tocnost `0.9111` i F1 `0.9117`, vrlo blizu
`LinearSVC` rezultatu.

Prednost `SGDClassifier` modela je skalabilnost. Kada bi skup imao milijune
recenzija, stohasticko treniranje moglo bi biti prakticnije od drugih postupaka.
U ovom projektu skup je dovoljno malen da `LinearSVC` radi bez problema, ali
SGD rezultat potvrduje da problem dobro odgovara linearnom margin-based
pristupu.

### 6.2.4 PassiveAggressiveClassifier

![Slika 10. `PassiveAggressiveClassifier`: online margin-based linearni model
ostaje pasivan kada je primjer dobro klasificiran, a agresivno azurira tezine
kada primjer krsi marginu.](figures/generated/classifier_cards/passive_aggressive.png){width=95%}

`PassiveAggressiveClassifier` radi online azuriranja i mijenja model uglavnom
kad je trenutna klasifikacija pogresna ili nedovoljno sigurna. Postigao je
tocnost `0.9062`. Njegov naziv opisuje postupak ucenja: model ostaje "passive"
kada je predikcija dobra, a "aggressive" kada treba ispraviti gresku.

Na staticnom IMDb skupu rezultat je malo slabiji od `LinearSVC` i
`SGDClassifier`, ali i dalje prelazi `0.90` tocnosti. Matrica zabune je gotovo
uravnotezena, s 468 lazno pozitivnih i 470 lazno negativnih gresaka, pa model
nema izrazitu pristranost prema jednoj klasi.

## 6.3 Stablima temeljeni ensemble modeli

`RandomForestClassifier` i `ExtraTreesClassifier` grade ansamble stabala
odluke. Takvi modeli su vrlo korisni na tablicnim podacima, ali na rijetkim
TF-IDF vektorima cesto nisu najbolji izbor. Imaju mnogo potencijalnih splitova
i mogu biti sporiji ili manje ucinkoviti od linearnih modela.

### 6.3.1 RandomForestClassifier

![Slika 11. `RandomForestClassifier`: vise stabala trenira se na bootstrap
uzorcima, svako stablo glasa, a konacna klasa dolazi iz vecinskog glasanja ili
prosjeka vjerojatnosti.](figures/generated/classifier_cards/random_forest.png){width=95%}

`RandomForestClassifier` je postigao tocnost `0.8633`, slabije od Bayesovih i
linearnih modela. To je korisna lekcija: model koji je jak na tablicnim podacima
ne mora biti najbolji za rijetke tekstualne znacajke. Stabla moraju birati
splitove po pojedinim znacajkama, a TF-IDF prostor ima mnogo rijetkih stupaca.

Signal sentimenta je rasprsen preko velikog broja rijeci i fraza, pa linearni
model koji koristi sve znacajke odjednom bolje odgovara problemu. Random forest
zato ostaje vazan dio propisane usporedbe, ali ne i glavni kandidat za najbolji
rezultat.

### 6.3.2 ExtraTreesClassifier

![Slika 12. `ExtraTreesClassifier`: ansambl ekstremno randomiziranih stabala
koristi dodatnu slucajnost u odabiru znacajki i pragova splitova, sto ga
razlikuje od random foresta.](figures/generated/classifier_cards/extra_trees.png){width=95%}

`ExtraTreesClassifier` je postigao tocnost `0.8749`, bolje od random foresta,
ali i dalje slabije od linearnih modela. Veca randomizacija u splitovima i
ansamblu ovdje pomaze, no ne mijenja osnovnu cinjenicu da stabla nisu prirodan
prvi izbor za TF-IDF tekst.

U usporedbi je ipak koristan jer pokazuje razliku izmedu dva slicna bagging
pristupa. Extra Trees je bio bolji od Random Forest modela, ali oba ostaju
ispod `MultinomialNB`, `LogisticRegression`, `SGDClassifier` i `LinearSVC`
rezultata.

### 6.3.3 XGBoostClassifier

![Slika 13. `XGBoostClassifier`: gradient boosting gradi stabla sekvencijalno i
svako novo stablo pokusava ispraviti preostale pogreske; na ovom rijetkom
TF-IDF splitu model je imao najvise lazno pozitivnih pogresaka medu obaveznim
modelima.](figures/generated/classifier_cards/xgboost_classifier.png){width=95%}

`XGBoostClassifier` i `LightGBMClassifier` su gradient boosting modeli. Oni
grade niz stabala gdje svako sljedece stablo pokusava ispraviti pogreske
prethodnih. U ovom projektu su ukljuceni jer su izricito navedeni u zadatku i
jer predstavljaju jaku skupinu modela za mnoge probleme strojnog ucenja.

`XGBoostClassifier` je postigao tocnost `0.8509`, najslabiju medu obaveznim
modelima. To ne znaci da je XGBoost opcenito slab algoritam. Ovdje je problem
kombinacija boosting stabala i vrlo rijetkog tekstualnog prostora. Matrica
zabune pokazuje 898 lazno pozitivnih predikcija, sto znaci da je model cesce
oznacavao recenzije kao pozitivne nego sto je trebalo.

### 6.3.4 LightGBMClassifier

![Slika 14. `LightGBMClassifier`: LightGBM koristi ucinkovit histogram split
search i leaf-wise rast stabala; bio je najbolji tree/boosting model u projektu,
ali nije dostigao linearne TF-IDF modele po tocnosti.](figures/generated/classifier_cards/lightgbm_classifier.png){width=95%}

`LightGBMClassifier` je najbolji stablima temeljeni model u projektu. Postigao
je tocnost `0.8918`, F1 `0.8925` i log-loss `0.2617`. Po tocnosti ne dostize
linearne modele, ali ima vrlo dobar probabilisticki izlaz u ovom eksperimentu.

Zbog toga je LightGBM zanimljiv clan soft voting ensemblea. Iako nije najbolji
po tvrdim labelama, njegove vjerojatnosti mogu biti korisne kada se prosjecuju s
drugim probabilistickim modelima. To je dobar primjer zasto nije dovoljno
gledati samo accuracy; razlicite metrike mogu naglasiti razlicite kvalitete
modela.

# 7. Validacija i pretraga hiperparametara

Svi modeli su evaluirani na istom izdvojenom testnom skupu od 10 000 recenzija.
Za zahtjev `RandomizedSearchCV (CV)` izveden je zaseban tuned eksperiment za
`MultinomialNB`. Pretraga je koristila:

- `RandomizedSearchCV`,
- `n_iter=10`,
- `cv=3`,
- metriku za odabir `f1`,
- stratificirane foldove preko trening dijela skupa.

Najbolji pronadeni parametri bili su:

- `classifier__alpha=0.5`,
- `tfidf__max_df=0.9`,
- `tfidf__max_features=80000`,
- `tfidf__min_df=3`,
- `tfidf__ngram_range=(1, 2)`,
- `tfidf__sublinear_tf=True`.

Najbolji srednji F1 rezultat u unakrsnoj validaciji bio je `0.8850`. Na testnom
skupu tuned `MultinomialNB` postigao je tocnost `0.8854`, sto je malo bolje od
osnovnog `MultinomialNB` rezultata `0.8841`.

Ovaj rezultat pokazuje dvije stvari. Prvo, zahtijevani `RandomizedSearchCV` i
CV postupak su stvarno provedeni. Drugo, za ovaj konkretni skup najveci skok u
performansama nije dosao od finog podesavanja Bayesovog modela, nego od izbora
modela: linearni modeli nad istim TF-IDF znacajkama rade znatno bolje.

![Slika 15. Koncept `RandomizedSearchCV` postupka: uzorkuje se vise kombinacija
hiperparametara, svaka se ocjenjuje kroz foldove unakrsne validacije, a bira se
kombinacija s najboljim srednjim rezultatom.](figures/generated/randomized_search_cv.png){width=95%}

# 8. Rezultati klasicnih TF-IDF modela

Tablice 1 i 2 prikazuju glavne rezultate za propisane klasicne modele.
Ukljucen je i tuned `MultinomialNB` redak jer pokazuje rezultat
`RandomizedSearchCV` postupka. Tablice su podijeljene zbog citljivosti: prva
prikazuje osnovne klasifikacijske metrike, a druga metrike koje ovise o
rangiranju ili probabilistickim izlazima.

![Slika 16. Sazetak najboljih rezultata na testnom skupu od 10 000 recenzija:
najbolji klasicni model je `LinearSVC`, najbolji ensemble po tocnosti je hard
voting, najbolji probabilisticki ensemble je soft voting, a najbolji dodatni
transformer je `DeBERTa-v3-small`.](figures/generated/results_summary.png){width=95%}

| Model | ACC | BACC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MultinomialNB | 0.8841 | 0.8841 | 0.8808 | 0.8884 | 0.8846 | 0.7682 |
| MultinomialNB tuned | 0.8854 | 0.8854 | 0.8842 | 0.8870 | 0.8856 | 0.7708 |
| ComplementNB | 0.8841 | 0.8841 | 0.8808 | 0.8884 | 0.8846 | 0.7682 |
| LogisticRegression | 0.9095 | 0.9095 | 0.9044 | 0.9158 | 0.9101 | 0.8191 |
| LinearSVC | 0.9150 | 0.9150 | 0.9135 | 0.9168 | 0.9152 | 0.8300 |
| SGDClassifier | 0.9111 | 0.9111 | 0.9052 | 0.9184 | 0.9117 | 0.8223 |
| PassiveAggressive | 0.9062 | 0.9062 | 0.9064 | 0.9060 | 0.9062 | 0.8124 |
| RandomForest | 0.8633 | 0.8633 | 0.8673 | 0.8578 | 0.8625 | 0.7266 |
| ExtraTrees | 0.8749 | 0.8749 | 0.8820 | 0.8656 | 0.8737 | 0.7499 |
| XGBoost | 0.8509 | 0.8509 | 0.8307 | 0.8814 | 0.8553 | 0.7031 |
| LightGBM | 0.8918 | 0.8918 | 0.8865 | 0.8986 | 0.8925 | 0.7837 |

| Model | ROC-AUC | PR-AUC | Log-loss |
| --- | ---: | ---: | ---: |
| MultinomialNB | 0.9531 | 0.9513 | 0.3135 |
| MultinomialNB tuned | 0.9554 | 0.9537 | 0.2934 |
| ComplementNB | 0.9531 | 0.9513 | 0.3135 |
| LogisticRegression | 0.9710 | 0.9702 | 0.2707 |
| LinearSVC | 0.9720 | 0.9708 | n/a |
| SGDClassifier | 0.9714 | 0.9707 | n/a |
| PassiveAggressive | 0.9656 | 0.9638 | n/a |
| RandomForest | 0.9376 | 0.9299 | 0.4554 |
| ExtraTrees | 0.9466 | 0.9398 | 0.4342 |
| XGBoost | 0.9355 | 0.9337 | 0.3565 |
| LightGBM | 0.9605 | 0.9599 | 0.2617 |

Najbolji klasicni rezultat postigao je `LinearSVC`. To je ocekivano za ovakav
problem: TF-IDF prostor je visokodimenzionalan i rijedak, a linearna granica
odluke je dovoljno fleksibilna za mnogo tekstualnih klasifikacijskih zadataka.
`SGDClassifier` je vrlo blizu, sto dodatno potvrduje da linearni margin-based
pristupi dobro odgovaraju ovom prikazu podataka.

Bayesovi modeli su slabiji od linearnih modela, ali su i dalje vrlo dobri s
obzirom na jednostavnost. `LightGBMClassifier` je najbolji medu boosting/tree
modelima, ali ne dostize linearne modele. `XGBoostClassifier` je ovdje najslabiji
od propisanih modela po tocnosti, sto pokazuje da jaki tablicni modeli nisu
automatski najbolji izbor za rijetke tekstualne vektore.

## 8.1 Matrice zabune za klasicne modele

U svim matricama redci su stvarne klase, a stupci predikcije. Klase su redom
`negative` i `positive`, pa su stupci u tablici:

- TN: negativna recenzija predvidena kao negativna,
- FP: negativna recenzija predvidena kao pozitivna,
- FN: pozitivna recenzija predvidena kao negativna,
- TP: pozitivna recenzija predvidena kao pozitivna.

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| MultinomialNB | 4399 | 601 | 558 | 4442 |
| MultinomialNB tuned | 4419 | 581 | 565 | 4435 |
| ComplementNB | 4399 | 601 | 558 | 4442 |
| LogisticRegression | 4516 | 484 | 421 | 4579 |
| LinearSVC | 4566 | 434 | 416 | 4584 |
| SGDClassifier | 4519 | 481 | 408 | 4592 |
| PassiveAggressiveClassifier | 4532 | 468 | 470 | 4530 |
| RandomForestClassifier | 4344 | 656 | 711 | 4289 |
| ExtraTreesClassifier | 4421 | 579 | 672 | 4328 |
| XGBoostClassifier | 4102 | 898 | 593 | 4407 |
| LightGBMClassifier | 4425 | 575 | 507 | 4493 |

`LinearSVC` ima 850 pogresaka na 10 000 testnih primjera. Od toga je 434 lazno
pozitivnih i 416 lazno negativnih. Pogreske su relativno uravnotezene izmedu
klasa, sto je u skladu s uravnotezenim testnim skupom i slicnim vrijednostima
precision i recall metrika.

![Slika 17. Usporedba broja pogresaka za najbolji klasicni model, najbolji
ensemble po tocnosti i najbolji dodatni transformer. Slika pokazuje da
transformer smanjuje ukupni broj pogresaka, ali ostaje oznacen kao dodatno
prosirenje.](figures/generated/error_counts.png){width=95%}

# 9. Zavrsni ensemble modeli

Zadatak trazi zavrsni `VotingClassifier` ili `StackingClassifier`. U projektu
su implementirane tri ensemble varijante:

- soft `VotingClassifier`,
- hard `VotingClassifier`,
- prefit `StackingClassifier`.

Soft voting kombinira vjerojatnosti modela koji ih mogu dati. Prednost mu je
sto zadrzava probabilisticki izlaz, pa je moguce racunati log-loss, ROC-AUC i
PR-AUC na standardan nacin. Hard voting kombinira tvrde glasove i bira klasu s
vecinom glasova. On je malo bolji po tocnosti u ovom eksperimentu, ali nema
standardni probabilisticki izlaz. Stacking uci dodatni meta-model nad izlazima
baznih modela.

![Slika 18. Usporedba ensemble arhitektura: soft voting prosjecuje
vjerojatnosti, hard voting koristi vecinsko glasanje, a prefit stacking koristi
spremljene bazne modele i cached meta-znacajke za meta-model.](figures/generated/ensemble_architectures_project.png){width=95%}

| Model | ACC | BACC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Voting soft | 0.9095 | 0.9095 | 0.9036 | 0.9168 | 0.9102 | 0.8191 |
| Voting hard | 0.9107 | 0.9107 | 0.9059 | 0.9166 | 0.9112 | 0.8215 |
| Stacking prefit | 0.9099 | 0.9099 | 0.9077 | 0.9126 | 0.9101 | 0.8198 |

| Model | ROC-AUC | PR-AUC | Log-loss |
| --- | ---: | ---: | ---: |
| Voting soft | 0.9707 | 0.9702 | 0.2654 |
| Voting hard | n/a | n/a | n/a |
| Stacking prefit | 0.9686 | 0.9669 | 0.4079 |

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| VotingClassifier soft | 4511 | 489 | 416 | 4584 |
| VotingClassifier hard | 4524 | 476 | 417 | 4583 |
| StackingClassifier prefit | 4536 | 464 | 437 | 4563 |

Kao zavrsni model u smislu zahtjeva kolegija moze se uzeti soft
`VotingClassifier`, jer ispunjava ensemble zahtjev i daje sve probabilisticke
metrike. Hard voting je takoder implementiran i ima nesto vecu tocnost, ali ne
daje log-loss bez dodatne kalibracije. Stacking model nije nadmasio voting ni
najbolji pojedinacni linearni model.

Vazno je da ensemble nije automatski najbolji model. U ovom projektu najbolji
pojedinacni klasicni model (`LinearSVC`) nadmasuje sve klasicne ensemble
varijante. To nije kontradikcija zadatka: zadatak trazi da se ensemble slozi,
ali rezultati pokazuju da za ovaj skup i ove postavke jednostavniji linearni
model ostaje najbolji klasicni izbor.

# 10. Dodatno istrazivanje: neuronski i transformerski modeli

Nakon obaveznog TF-IDF dijela isproban je dodatni smjer: mali transformerski
model od nule i fino ugadanje unaprijed naucenih modela. Ovaj dio nije zamjena
za propisane klasifikatore. U izvjestaju je ukljucen zato sto pokazuje granicu
klasicnih metoda i zato sto je analiza sentimenta danas cesto rjesavana
unaprijed naucenim jezicnim modelima. Eksperimenti su vodeni kao mala
istrazivacka iteracija: prvo se provjerava moze li vrlo mali transformer sam
nauciti jezik zadatka, zatim se mijenjaju vokabular, pooling, scheduler i
duljina konteksta, a na kraju se isti problem usporeduje s predtreniranim
modelima.

## 10.1 Transformer treniran od nule

![Slika 19. Mali transformer treniran od nule: BPE vokabular od 10 000 tokena,
embedding dimenzija 24, cetiri transformer encoder sloja, mean pooling i
klasifikacijska glava za pozitivan ili negativan sentiment.](figures/generated/tiny_transformer_from_scratch.png){width=95%}

Prvi cilj bio je namjerno strog: transformer ne smije koristiti predtreniranu
bazu, nego mora nauciti reprezentaciju iz samih IMDb oznaka. Pocetna MLX
varijanta imala je BPE vokabular od samo 256 tokena, `max_length=128`,
`d_model=16`, cetiri encoder sloja i oko 13 000 parametara. Iako se takav model
brzo trenirao, testna tocnost bila je samo `0.6385`. Taj rezultat je pokazao da
je model previse informacijski ogranicen: s vrlo malim vokabularom i kratkim
kontekstom veci dio recenzije se ili jako fragmentira ili odsijece.

Zato je napravljena druga serija eksperimenata. BPE tokenizer povecan je na
10 000 tokena i spremljen kao cache, kako se ne bi ponovno ucio pri svakom
pokretanju. Uveden je mean pooling preko tokena, dropout `0.1`, AdamW optimizator
i warmup-cosine raspored stope ucenja. Najbolji model treniran od nule imao je
`d_model=24`, cetiri encoder sloja, cetiri attention glave, `ff_dim=48`,
`max_length=512` i `263 306` parametara. Trening je izveden u MLX-u s
mikro-batchevima od 4 primjera i efektivnim batchem 16. Najbolja validacijska
tocnost bila je `0.8935`, a testna tocnost `0.8943`.

Ovaj rezultat je znatno bolji od pocetne sitne varijante, ali i dalje ne
nadmasuje `LinearSVC` s TF-IDF znacajkama. To je bitan zakljucak: arhitektura
transformera sama po sebi ne jamci bolji rezultat ako model nema dovoljno
predtreniranja ili dodatnog tekstualnog signala. Na udaljenoj RTX 3090 instanci
isproban je i PyTorch/CUDA ekvivalent radi provjere brzine i duljeg konteksta.
Varijanta s kontekstom 512 trenirala je oko dvije minute i postigla tocnost
`0.8733`, a varijanta s kontekstom 1024 oko 3.7 minuta i tocnost `0.8840`.
Dulji kontekst je pomogao, ali nije zatvorio razliku prema najboljim linearnim
TF-IDF modelima.

## 10.2 Fine-tuning predtreniranih transformera

Zatim su fino ugodena dva unaprijed naucena modela:

- `distilbert-base-uncased`,
- `microsoft/deberta-v3-small`.

![Slika 20. Fine-tuning predtreniranog transformera: model prvo uci opce jezicne
reprezentacije na velikim korpusima, a zatim se prilagodava IMDb oznakama
sentimenta pomocu klasifikacijske glave.](figures/generated/pretrained_transformer_finetuning.png){width=95%}

Treniranje je izvedeno na udaljenoj Vast.ai RTX 3090 instanci, ne na lokalnom
Mac racunalu. To je vazno jer su transformerski eksperimenti racunalno znatno
zahtjevniji od TF-IDF modela.

Kod fine-tuninga se vise ne uci jezicna reprezentacija od nule. Predtrenirani
model vec ima naucene obrasce engleskog jezika, sintakse, konteksta i
semantickih slicnosti iz velikih korpusa. IMDb trening skup tada sluzi za
prilagodbu klasifikacijske glave i unutarnjih reprezentacija na konkretan
zadatak pozitivnog ili negativnog sentimenta. Koristen je AdamW s
`learning_rate=2e-5`, `weight_decay=0.01`, warmup omjerom `0.06` i mixed
precision treniranjem na CUDA uredaju.

Nije posebno mjeren testni accuracy predtreniranih BERT modela prije
fine-tuninga. To bi bilo lose definirano za ovu implementaciju: checkpoint
`distilbert-base-uncased` ili `microsoft/deberta-v3-small` ima predtreniranu
jezicnu bazu, ali klasifikacijska glava u `AutoModelForSequenceClassification`
nije IMDb sentiment model dok se ne prilagodi na oznake. Takva "prije treninga"
tocnost mjerila bi uglavnom slucajno inicijaliziranu glavu, a ne stvarnu
sposobnost predtreniranog jezicnog modela za sentiment.

U sljedecoj tablici "Start" oznacava je li model krenuo od random
inicijalizacije ili od predtrenirane jezicne baze s novom klasifikacijskom
glavom. `Val@1` je validacijska tocnost nakon prve epohe fine-tuninga.

| Model | Start | ACC prije FT | Val@1 | Best val | Test ACC | Greske |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | scratch | n/a | 0.8613 | 0.8935 | 0.8943 | 1057 |
| DistilBERT | pretrained + nova glava | nije mjereno | 0.9220 | 0.9340 | 0.9369 | 631 |
| DeBERTa | pretrained + nova glava | nije mjereno | 0.9493 | 0.9565 | 0.9564 | 436 |

| Model | Prec. | Rec. | F1 | ROC-AUC | PR-AUC | Log-loss | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 0.8796 | 0.9136 | 0.8963 | 0.9603 | 0.9592 | 0.2789 | 0.7892 |
| DistilBERT | 0.9296 | 0.9454 | 0.9374 | 0.9840 | 0.9831 | 0.1888 | 0.8739 |
| DeBERTa | 0.9521 | 0.9612 | 0.9566 | 0.9895 | 0.9886 | 0.1561 | 0.9128 |

Najbolji ukupni model je `microsoft/deberta-v3-small`. On ima 436 pogresaka na
10 000 testnih primjera, dok `LinearSVC` ima 850 pogresaka. Razlika je velika,
ali dolazi iz unaprijed naucenog jezicnog znanja, a ne samo iz arhitekture. To
je glavna pouka dodatnog eksperimenta: na ogranicenom oznacenom skupu, model s
predtreniranim razumijevanjem jezika moze puno bolje generalizirati nego model
ucen od nule.

# 11. Rasprava

Rezultati pokazuju nekoliko jasnih obrazaca.

Prvo, TF-IDF i linearni modeli su vrlo jaki za ovaj zadatak. `LinearSVC`,
`SGDClassifier` i `LogisticRegression` svi prelaze tocnost od `0.90`. To je
dobar rezultat s obzirom na jednostavnost prikaza i cinjenicu da se ne koristi
nikakvo duboko jezicno predznanje.

Drugo, naivni Bayesovi modeli su korisni kao brz baseline. Njihova tocnost oko
`0.884` nije najbolja, ali je dovoljno visoka da pokaze kako se veliki dio
signala nalazi u samim rijecima i frazama. Tuniranje `MultinomialNB` modela
donijelo je malo poboljsanje, ali nije promijenilo ukupni poredak modela.

Trece, modeli temeljeni na stablima nisu optimalni za ovaj TF-IDF prikaz.
`LightGBMClassifier` je najbolji iz te skupine, ali i dalje zaostaje za
linearnim modelima. Razlog je vjerojatno priroda znacajki: TF-IDF matrica je
vrlo rijetka i visokodimenzionalna, a linearni modeli mogu izravno iskoristiti
takvu strukturu.

Cetvrto, ensemble modeli nisu automatski bolji od najboljeg pojedinacnog
modela. Voting i stacking modeli stabilno rade, ali ne nadmasuju `LinearSVC`.
To je korisna prakticna lekcija: ensemble je tehnika koja moze pomoci kada
bazni modeli imaju razlicite pogreske, ali nije jamstvo boljeg rezultata.

Peto, dodatni transformerski eksperimenti pokazuju veliku razliku izmedu modela
ucenog od nule i modela s predtreniranjem. Mali transformer od nule ne uspijeva
nadmasiti TF-IDF baseline, dok DeBERTa-v3-small znatno nadmasuje sve klasicne
modele. Time se vidi vrijednost transfer learninga u obradi prirodnog jezika.

# 12. Implementacijska organizacija

Projekt nije raden kao notebook-first projekt, nego kao Python workflow. To je
namjerno, jer je lakse pratiti svaki model kao zaseban eksperiment i ponovno
pokretati pojedine dijelove. Svaki klasifikator ima vlastitu mapu u
`classifiers/`, obicno s `run.py` skriptom i `README.md` datotekom.

Zajednicke funkcije nalaze se u `src/sentiment/`:

- `data.py` ucitava Kaggle CSV i priprema splitove,
- `metrics.py` racuna metrike i matricu zabune,
- `artifacts.py` sprema JSON i tekstualne artefakte,
- `paths.py` definira standardne putanje.

Takva organizacija smanjuje ponavljanje. Pojedinacni modeli mogu se razlikovati
u hiperparametrima i nacinu treniranja, ali dijele isti nacin ucitavanja
podataka, istu evaluaciju i isti oblik izlaznih artefakata. To je vazno za
postenu usporedbu: ako bi svaki model imao drugaciji split ili drugacije
racunanje metrika, rezultati ne bi bili izravno usporedivi.

Za svaki eksperiment sprema se `run_config.json`. Ta datoteka je posebno vazna
jer dokumentira sto je tocno pokrenuto: parametre TF-IDF vektorizatora, broj
redaka, split, model i dostupnost probabilistickih izlaza. Brojcani rezultati u
izvjestaju nisu pisani napamet, nego su preuzeti iz `metrics.json` i
`confusion_matrix.json` artefakata.

# 13. Analiza gresaka i moguci sljedeci koraci

Ovaj projekt ne ukljucuje duboku kvalitativnu analizu pojedinacnih pogresno
klasificiranih recenzija, ali se iz matrica zabune mogu izvuci korisni
zakljucci. Kod najboljeg klasicnog modela, `LinearSVC`, broj lazno pozitivnih
i lazno negativnih primjera gotovo je jednak. To znaci da model nema ocit
pomak prema jednoj klasi. Kod `XGBoostClassifier` modela vidi se drugacija
slika: broj lazno pozitivnih predikcija je znatno veci, pa model cesce
prepoznaje recenzije kao pozitivne.

U stvarnom sustavu analiza gresaka bila bi sljedeci korak. Posebno bi bilo
korisno pregledati:

- sarkasticne recenzije,
- recenzije s mijesanim sentimentom,
- recenzije koje pozitivno govore o jednom aspektu filma, ali negativno o
  ukupnom dojmu,
- vrlo kratke recenzije,
- recenzije s rijecima koje imaju drugacije znacenje ovisno o kontekstu.

TF-IDF modeli tesko razumiju ironiju, negaciju preko duzeg konteksta i
semanticke odnose. Bigrami pomazu kod kratkih fraza poput "not good", ali ne
rjesavaju cijeli problem. To objasnjava zasto predtrenirani transformer moze
biti bolji: on dolazi s vec naucenim jezicnim reprezentacijama i moze bolje
iskoristiti kontekst.

Moguci nastavci projekta bili bi:

- dodatna kalibracija `LinearSVC` modela,
- tuning `LogisticRegression` i `LightGBMClassifier` modela,
- optimizacija tezina soft voting ensemblea,
- evaluacija na Stanford ACL IMDB splitu kao dodatnom vanjskom testu,
- interpretacija znacajki za linearne modele,
- analiza pogresno klasificiranih primjera.

# 14. Reproducibilnost

Projekt je organiziran kao Python repozitorij. Glavni kod je u mapama:

- `src/sentiment/` za zajednicke funkcije ucitavanja podataka, metrika i
  putanja,
- `classifiers/` za pojedinacne modele,
- `outputs/` za generirane rezultate,
- `reports/` za izvjestajne materijale.

Primjer pokretanja jednog obaveznog modela:

```bash
PYTHONPATH=. python -m classifiers.linear_svc.run
```

Primjer pokretanja tuned Bayesovog modela s `RandomizedSearchCV`:

```bash
PYTHONPATH=. python -m classifiers.multinomial_nb.run \
  --tune \
  --n-iter 10 \
  --cv 3 \
  --output-dir outputs/baselines/multinomial_nb_tuned_n10_cv3
```

Primjer pokretanja soft voting ensemble modela:

```bash
PYTHONPATH=. python -m classifiers.voting_classifier.run
```

Za svaki model spremaju se:

- `metrics.json`,
- `confusion_matrix.json`,
- `classification_report.txt`,
- `run_config.json`,
- po potrebi trenirani model.

Transformerski eksperimenti su racunalno zahtjevniji i nisu potrebni za osnovno
pokretanje projekta. Pokretani su na GPU instanci, a rezultati su prepisani u
izvjestaj.

# 15. Ogranicenja

Glavno ogranicenje projekta je to sto je primarna evaluacija napravljena na
jednom splitu Kaggle IMDb skupa. Iako je split stratificiran i velik, dodatna
evaluacija na potpuno odvojenom izvoru mogla bi dati jos jacu procjenu
generalizacije.

Drugo ogranicenje je opseg hiperparametarske pretrage. `RandomizedSearchCV` je
proveden i dokumentiran za `MultinomialNB`, ali nisu svi modeli duboko tunirani.
To je praktican kompromis: deset modela, ensemble varijante i dodatni
transformerski eksperimenti vec daju velik usporedni prostor. Za maksimalno
optimiranje klasicnih modela bilo bi korisno prosiriti pretragu na
`LogisticRegression`, `LinearSVC`, `LightGBMClassifier` i ensemble tezine.

Trece ogranicenje je log-loss kod modela bez probabilistickog izlaza. Umjesto
da se ta metrika umjetno racuna iz tvrdih predikcija, u izvjestaju je oznaceno
da nije dostupna. Alternativa bi bila kalibracija modela pomocu
`CalibratedClassifierCV`, ali to bi promijenilo modele i dodalo jos jedan sloj
treniranja.

# 16. Zakljucak

Projekt ispunjava osnovne zahtjeve teme 7: koristi IMDb recenzije, TF-IDF
prikaz, deset propisanih klasifikatora, `RandomizedSearchCV` s unakrsnom
validacijom, standardne metrike i zavrsne Voting/Stacking ensemble modele.

Najbolji klasicni model je `LinearSVC` s tocnosti `0.9150`. Najbolji obavezni
ensemble je hard `VotingClassifier` po tocnosti, dok je soft `VotingClassifier`
prikladniji kao zavrsni probabilisticki ensemble jer daje log-loss i
vjerojatnosti. Ensemble modeli nisu nadmasili `LinearSVC`, sto pokazuje da je
za TF-IDF tekstualne znacajke dobro odabran linearni model vrlo jak baseline.

Dodatni eksperimenti s transformerima pokazuju da moderno predtreniranje moze
znatno poboljsati rezultat. `DeBERTa-v3-small` postize tocnost `0.9564`, sto je
znatno vise od klasicnih TF-IDF modela. Ipak, za zahtjeve kolegija kljucan je
klasicni TF-IDF dio, a transformerski dio treba citati kao dodatno
istrazivanje.

# Literatura i izvori

- Materijali kolegija: `PUI - Osnovne Informacije o kolegiju.pdf`.
- Kaggle: IMDb 50K Movie Reviews.
- Stanford AI Lab: IMDb Large Movie Review Dataset.
- scikit-learn dokumentacija za `TfidfVectorizer`, linearne modele, naive Bayes
  modele, ensemble modele i metrike.
- scikit-learn example: "Classification of text documents using sparse
  features", koristan kao struktura za usporedbu vise klasifikatora nad istim
  tekstualnim vektorskim prikazom.
- Banerjee et al. (2026): "From TF-IDF to Transformers: A Comparative Study of
  Sentiment Analysis Methods in Python", arXiv:2605.07811.
- "From Machine Learning to Deep Learning: Enhancing IMDb Movie Review
  Sentiment Analysis" (2026), arXiv:2605.22003.
- XGBoost dokumentacija.
- LightGBM dokumentacija.
- Hugging Face Transformers dokumentacija za DistilBERT i DeBERTa modele.

# Dodatak A: Artefakti u repozitoriju

Glavni lokalni artefakti:

- `outputs/baselines/*/metrics.json`
- `outputs/baselines/*/confusion_matrix.json`
- `outputs/ensemble/*/metrics.json`
- `outputs/ensemble/*/confusion_matrix.json`
- `outputs/transformer/*/metrics.json`
- `reports/final_report/results_table.csv`
- `reports/final_report/results_tables.md`

Ovi artefakti su izvor brojcanih rezultata prikazanih u izvjestaju.

# Dodatak B: Komande za ponavljanje eksperimenata

Sljedeca tablica navodi komande za glavne pokrete. Komande se izvode iz korijena
repozitorija. Ako paket nije instaliran u editable modu, koristi se
`PYTHONPATH=.` prefiks kao u primjerima.

| Eksperiment | Komanda | Glavni izlaz |
| --- | --- | --- |
| MultinomialNB baseline | `PYTHONPATH=. python -m classifiers.multinomial_nb.run` | `outputs/baselines/multinomial_nb` |
| MultinomialNB tuned | `PYTHONPATH=. python -m classifiers.multinomial_nb.run --tune --n-iter 10 --cv 3 --output-dir outputs/baselines/multinomial_nb_tuned_n10_cv3` | `outputs/baselines/multinomial_nb_tuned_n10_cv3` |
| ComplementNB baseline | `PYTHONPATH=. python -m classifiers.complement_nb.run` | `outputs/baselines/complement_nb` |
| LogisticRegression baseline | `PYTHONPATH=. python -m classifiers.logistic_regression.run` | `outputs/baselines/logistic_regression` |
| LinearSVC baseline | `PYTHONPATH=. python -m classifiers.linear_svc.run` | `outputs/baselines/linear_svc` |
| SGDClassifier baseline | `PYTHONPATH=. python -m classifiers.sgd_classifier.run` | `outputs/baselines/sgd_classifier` |
| PassiveAggressive baseline | `PYTHONPATH=. python -m classifiers.passive_aggressive.run` | `outputs/baselines/passive_aggressive` |
| RandomForest baseline | `PYTHONPATH=. python -m classifiers.random_forest.run` | `outputs/baselines/random_forest` |
| ExtraTrees baseline | `PYTHONPATH=. python -m classifiers.extra_trees.run` | `outputs/baselines/extra_trees` |
| XGBoost baseline | `PYTHONPATH=. python -m classifiers.xgboost_classifier.run` | `outputs/baselines/xgboost_classifier` |
| LightGBM baseline | `PYTHONPATH=. python -m classifiers.lightgbm_classifier.run` | `outputs/baselines/lightgbm_classifier` |
| Soft VotingClassifier | `PYTHONPATH=. python -m classifiers.voting_classifier.run` | `outputs/ensemble/voting_classifier` |
| Hard VotingClassifier | `PYTHONPATH=. python -m classifiers.hard_voting_classifier.run` | `outputs/ensemble/hard_voting_classifier` |
| Prefit StackingClassifier | `PYTHONPATH=. python -m classifiers.stacking_classifier_prefit.run` | `outputs/ensemble/stacking_classifier_prefit` |

Svaki od ovih izlaznih direktorija sadrzi barem `metrics.json`,
`confusion_matrix.json`, `classification_report.txt` i `run_config.json`.
`outputs/` direktorij nije namijenjen predaji kao veliki generirani artefakt,
zato su zavrsni rezultati prepisani u `reports/final_report/results_table.csv`
i `reports/final_report/results_tables.md`.

# Dodatak C: Pregled konfiguracija klasicnih modela

Svi klasicni modeli koriste isti osnovni dataset, split i TF-IDF prikaz, osim
tuned `MultinomialNB` eksperimenta gdje je `RandomizedSearchCV` odabrao malo
drugacije TF-IDF parametre. Osnovni split je 40 000 trening recenzija i 10 000
testnih recenzija, s `random_state=42`.

| Model | Glavni parametri modela | Probabilisticki izlaz | Napomena |
| --- | --- | --- | --- |
| MultinomialNB | `alpha=1.0` | da | brzi tekstualni baseline |
| MultinomialNB tuned | `alpha=0.5`, tuned TF-IDF | da | `RandomizedSearchCV`, `n_iter=10`, `cv=3`, najbolji CV F1 `0.8850` |
| ComplementNB | `alpha=1.0` | da | isti rezultat kao MultinomialNB na uravnotezenom splitu |
| LogisticRegression | `C=1.0`, `solver=liblinear`, `max_iter=1000` | da | jak linearni probabilisticki baseline |
| LinearSVC | `C=1.0`, `dual=auto`, `max_iter=5000` | ne | najbolji klasicni model po tocnosti/F1 |
| SGDClassifier | `loss=hinge`, `alpha=0.0001`, `max_iter=1000` | ne | SVM-slican linearni model treniran SGD-om |
| PassiveAggressiveClassifier | `C=1.0`, `max_iter=1000` | ne | online margin-based model |
| RandomForestClassifier | `n_estimators=100`, `max_features=sqrt` | da | slabiji na rijetkim TF-IDF znacajkama |
| ExtraTreesClassifier | `n_estimators=100`, `max_features=sqrt` | da | bolji od random foresta, ali slabiji od linearnih modela |
| XGBoostClassifier | `n_estimators=200`, `max_depth=4`, `tree_method=hist` | da | boosting model s izrazenim FP greskama u ovom splitu |
| LightGBMClassifier | `n_estimators=200`, `num_leaves=31`, `learning_rate=0.1` | da | najbolji tree/boosting model i najnizi klasicni log-loss |

Osnovni TF-IDF parametri su `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`,
`max_features=50000`, `strip_accents="unicode"` i `sublinear_tf=True`. Tuned
`MultinomialNB` koristi `max_df=0.9`, `min_df=3`, `max_features=80000` i isti
raspon n-grama. Vokabular se uvijek uci samo na trening podacima.

# Dodatak D: Detalji ensemble modela

Soft `VotingClassifier` koristi tri probabilisticka clana:
`LogisticRegression`, `MultinomialNB` i `LightGBMClassifier`. Ovaj izbor je
praktican jer sva tri clana daju `predict_proba`, pa se mogu racunati
standardni log-loss, ROC-AUC i PR-AUC. Njegov log-loss `0.2654` bolji je od
log-loss vrijednosti same logisticke regresije (`0.2707`), iako mu tocnost nije
bolja od `LinearSVC` modela.

Hard `VotingClassifier` ukljucuje sve glavne klasicne modele, ukljucujuci i
tuned `MultinomialNB` varijantu. Ima najbolju ensemble tocnost `0.9107`, ali
nema standardni probabilisticki izlaz. U zasebnom README-u zabiljezene su i
vote-fraction vrijednosti kao gruba ordinalna mjera, no one se u glavnoj
tablici ne tretiraju kao standardni log-loss jer nisu kalibrirane
vjerojatnosti.

Prefit stacking koristi spremljene bazne modele i gradi meta-znacajke iz
njihovih izlaza. Za modele s `predict_proba` koristi probabilisticke izlaze, a
za `LinearSVC`, `SGDClassifier` i `PassiveAggressiveClassifier` koristi
`decision_function`. Meta-model je logisticka regresija. Ova implementacija je
prakticna jer izbjegava ponovno treniranje svih baznih modela, ali je u
izvjestaju jasno oznacena kao prefit/cached stacking, a ne kao skuplja
out-of-fold stacking procedura.

# Dodatak E: Detalji transformerskog prosirenja

Transformerski dio ima dvije razlicite svrhe. Mali transformer od nule sluzi
kao istrazivanje koliko se moze nauciti bez predtreniranja. Pretrained
transformeri sluze kao usporedba s modernijim NLP pristupom koji koristi
predznanje nauceno na mnogo vecim korpusima.

| Model | Parametri | Max length | Trening | Najbolja val tocnost | Test tocnost | Runtime |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Tiny MLX pocetni pokus | oko 13k | 128 | MLX, 256 BPE tokena | oko `0.646` | `0.6385` | oko 70 s |
| Tiny MLX najbolji od nule | 263k | 512 | MLX, 10k BPE, mean pooling, mikro-batch 4 | `0.8935` | `0.8943` | oko 17.1 min |
| Tiny Torch CUDA kontekst 1024 | 263k | 1024 | RTX 3090, fp16, batch 16 | `0.8742` | `0.8840` | oko 3.7 min |
| DistilBERT | 66.96M | 512 | RTX 3090, 3 epohe, batch 32 | `0.9340` | `0.9369` | oko 10.8 min |
| DeBERTa-v3-small | 141.90M | 512 | RTX 3090, 3 epohe, batch 16 | `0.9565` | `0.9564` | oko 25.9 min |

Za `DistilBERT` i `DeBERTa-v3-small` koristen je AdamW, `learning_rate=2e-5`,
`weight_decay=0.01`, warmup omjer `0.06` i mixed precision na CUDA uredaju.
`DistilBERT` je koristio batch 32, a `DeBERTa-v3-small` batch 16. Oba modela
koriste 36 000 primjera za treniranje, 4 000 za validaciju i 10 000 za test.

Najvazniji zakljucak je da arhitektura sama po sebi nije dovoljna. Mali
transformer od nule ne nadmasuje `LinearSVC`, dok predtrenirani DeBERTa model
znatno nadmasuje sve klasicne modele. Time se jasno vidi razlika izmedu
ucenja reprezentacije od nule na 40 000 recenzija i prijenosa znanja iz velikog
jezicnog predtreniranja.

# Dodatak F: Kontrolna lista zahtjeva

Ova kontrolna lista sluzi kao zavrsna provjera uskladenosti s temom 7.

| Zahtjev | Status u projektu | Dokaz |
| --- | --- | --- |
| Analiza sentimenta recenzija | ispunjeno | Kaggle IMDb recenzije, binarne oznake `positive`/`negative` |
| Bez rucnog oznacavanja | ispunjeno | koriste se vec oznaceni javni skupovi |
| TF-IDF prikaz | ispunjeno | `TfidfVectorizer` u klasicnim pipelineovima |
| Najmanje 10 klasifikatora | ispunjeno | 10 propisanih modela plus tuned NB redak |
| RandomizedSearchCV | ispunjeno | tuned `MultinomialNB`, `n_iter=10`, `cv=3` |
| Cross-validation | ispunjeno | 3-fold CV u tuned NB eksperimentu |
| ACC, BACC, precision, recall, F1 | ispunjeno | tablice rezultata u poglavlju 8 |
| ROC-AUC i PR-AUC | ispunjeno | racunato iz vjerojatnosti ili `decision_function` |
| MCC | ispunjeno | tablice rezultata u poglavlju 8 |
| Log-loss | ispunjeno gdje je matematicki dostupan | n/a objasnjen za modele bez kalibriranih vjerojatnosti |
| Matrice zabune | ispunjeno | poglavlja 8, 9 i 10 |
| Voting ili Stacking finalni model | ispunjeno | soft/hard voting i prefit stacking |
| Izvjestaj u prihvatljivom formatu | ispunjeno | `reports/final_report/analiza_sentimenta_recenzija.docx` |
| Prezentacija u prihvatljivom formatu | ispunjeno | `presentation/analiza_sentimenta_recenzija.pptx` |
| Upute za pokretanje | ispunjeno | `README.md` i poglavlje 14 |

Preostali kompromisi su namjerno navedeni u poglavlju ogranicenja: nije tuniran
svaki model, primarna evaluacija je jedan stratificirani Kaggle split, a
nekalibrirani margin-based modeli nemaju log-loss. Ti kompromisi ne mijenjaju
cinjenicu da su trazeni dijelovi projekta provedeni i dokumentirani.

# Dodatak G: Kratki plan usmene prezentacije

Prezentacija je zamisljena za 10-15 minuta. Cilj nije procitati cijeli
izvjestaj, nego jasno pokazati da su ispunjeni zahtjevi teme 7 i da su rezultati
dobiveni stvarnim eksperimentima.

## G.1 Uvod i zadatak

Prvi dio prezentacije treba u jednoj minuti reci sto je problem: binarna
klasifikacija IMDb recenzija na pozitivan i negativan sentiment. Treba odmah
naglasiti da nije bilo rucnog labeliranja i da su dataset izvori oni navedeni u
materijalima kolegija. Time se zatvara moguce pitanje je li tema zahtijevala
manualno oznacavanje podataka.

## G.2 Podaci i TF-IDF pipeline

Drugi dio treba objasniti 50 000 recenzija, stratificirani 80/20 split i
cinjenicu da je testni skup uravnotezen s 5 000 pozitivnih i 5 000 negativnih
recenzija. Kod TF-IDF-a treba reci da se vokabular uci iskljucivo na trening
skupu i da testni podaci ne smiju utjecati na izbor znacajki. Vizualno je
najbolje proci kroz tok: raw review, TF-IDF vokabular, sparse matrix,
klasifikator, sentiment.

## G.3 Modeli i metrika

Treba kratko grupirati modele umjesto objasnjavati svih deset jednako dugo:
Bayesovi modeli, linearni modeli, tree/bagging modeli i boosting modeli.
Najvaznija poruka je da su linearni modeli najprirodniji za rijedak
visokodimenzionalni TF-IDF prostor. Kod metrika treba naglasiti da nisu
prikazani samo accuracy i F1, nego i ROC-AUC, PR-AUC, MCC, log-loss i matrice
zabune.

## G.4 Rezultati i ensemble

Glavni rezultat obaveznog dijela je `LinearSVC` s tocnosti `0.9150`. Ensemble
dio treba prikazati iskreno: soft voting, hard voting i prefit stacking su
implementirani, ali ne nadmasuju najbolji pojedinacni linearni model. To nije
neuspjeh, nego rezultat eksperimenta. Zadatak trazi da se ensemble slozi, a ne
da on nuzno bude najbolji model.

## G.5 Transformer prosirenje

Transformer dio treba predstaviti kao dodatno istrazivanje. Mali transformer od
nule ne pobjeduje TF-IDF linearni model, sto je vazan negativan rezultat.
Predtrenirani modeli, posebno `DeBERTa-v3-small`, daju znatno bolji rezultat
jer koriste transfer learning. Treba jasno reci da transformer ne zamjenjuje
obavezni TF-IDF dio.

## G.6 Zakljucak

Zavrsna poruka: projekt ispunjava propisani TF-IDF dio, linearni modeli su
najbolji za ovaj prikaz podataka, ensemble je implementiran, a predtrenirani
transformeri pokazuju korist transfer learninga. Sljedeci koraci su kalibrirati
`LinearSVC` i provjeriti rezultat na Stanford ACL IMDb skupu.
