---
title: "Analiza sentimenta recenzija filmova"
subtitle: "Projekt iz kolegija Primjena umjetne inteligencije"
author: "Luka Ivanić"
date: "2026-06-04"
lang: hr-HR
---

# Sažetak

U ovom projektu razvijen je sustav za binarnu klasifikaciju sentimenta filmskih
recenzija. Ulaz u sustav je tekst recenzije, a izlaz je oznaka `positive` ili
`negative`. Projekt odgovara temi 7 iz materijala kolegija: "Analiza sentimenta
recenzija (tekst klasifikacija)". Prema zadatku, naglasak je na TF-IDF prikazu
teksta, usporedbi najmanje deset klasifikatora, korištenju
`RandomizedSearchCV` postupka s unakrsnom validacijom, računanju standardnih
metrika i izradi završnog `VotingClassifier` ili `StackingClassifier`
ensemble modela.

Primarni skup podataka u eksperimentima je Kaggle IMDb 50K Movie Reviews, s
50 000 označenih recenzija. Dodatno je preuzet i Stanford IMDb Large Movie
Review Dataset jer je naveden u materijalima kolegija kao drugi izvor za istu
temu. Nije bilo ručnog označavanja podataka: svi modeli uče iz već označenih
recenzija.

Najbolji klasični model bio je `LinearSVC` nad TF-IDF značajkama s točnosti
`0.9150` i F1 mjerom `0.9152` na izdvojenom testnom skupu. Završni ensemble
modeli su implementirani pomoću `VotingClassifier` i `StackingClassifier`.
Najbolji ensemble po točnosti bio je hard `VotingClassifier` s točnosti
`0.9107`, dok soft `VotingClassifier` daje i probabilističke metrike, uključivo
log-loss. Kao istraživačko proširenje, istrenirani su i transformerski modeli.
Najbolji ukupni rezultat postigao je fino ugođeni `microsoft/deberta-v3-small`
s točnosti `0.9564`, F1 mjerom `0.9566` i ROC-AUC vrijednosti `0.9895`.
Transformerski modeli nisu zamjena za obavezni TF-IDF dio, nego dodatna analiza
modernijeg pristupa istom problemu.

# 1. Uvod

Analiza sentimenta je zadatak automatskog prepoznavanja stava ili emocionalnog
tona u tekstu. U ovom projektu radi se o pojednostavljenoj, ali praktičnoj
varijanti problema: recenzija filma klasificira se kao pozitivna ili negativna.
Takav problem se pojavljuje u preporučnim sustavima, analizi korisničkih
komentara, praćenju reputacije proizvoda i automatskoj obradi velikih zbirki
tekstova.

Za razliku od numeričkih ili tabličnih problema, tekst nije izravno pogodan za
klasične algoritme strojnog učenja. Potrebno ga je pretvoriti u vektor
značajki. U zadatku je propisano korištenje TF-IDF prikaza. TF-IDF je
jednostavan, ali jak prikaz teksta: dokument se opisuje riječima i n-gramima
koji se pojavljuju u njemu, pri čemu se smanjuje utjecaj riječi koje se
pojavljuju u gotovo svim dokumentima.

Ovaj rad prvo obrađuje klasični zadani dio projekta:

- učitavanje IMDb recenzija,
- pretvorbu teksta u TF-IDF značajke,
- treniranje i usporedbu deset propisanih klasifikatora,
- računanje metrika,
- RandomizedSearchCV eksperiment s unakrsnom validacijom,
- završni Voting/Stacking ensemble.

Nakon toga prikazano je dodatno proširenje s neuronskim modelima. To proširenje
nije potrebno za osnovno ispunjenje zadatka, ali pomaže razumjeti koliko daleko
se može doći s modernim unaprijed naučenim jezičnim modelima u usporedbi s
klasičnim TF-IDF modelima.

# 2. Definicija zadatka

Prema dokumentu `PUI - Osnovne Informacije o kolegiju.pdf`, tema 7 je:

> Analiza sentimenta recenzija (tekst klasifikacija): TF-IDF + klasifikatori;
> usporediti modele i složiti ensemble (voting/stacking).

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
- završni `VotingClassifier` ili `StackingClassifier`.

U ovom projektu zadatak je interpretiran kao nadzirana binarna klasifikacija:

- ulaz: tekst jedne filmske recenzije,
- izlaz: negativan ili pozitivan sentiment.

# 3. Skup podataka

Materijali kolegija navode dva moguća izvora:

- Kaggle: IMDb 50K Movie Reviews,
- Stanford: IMDb Large Movie Review Dataset.

Eksperimenti u izvještaju koriste Kaggle IMDb 50K CSV jer je u njemu cijeli
skup dostupan u jednoj tablici s tekstom recenzije i oznakom sentimenta. Takav
oblik je pogodan za ponovljive eksperimente, izradu jedinstvenog train/test
splita i usporedbu više modela pod istim uvjetima.

Skup Kaggle IMDb 50K sadrži:

- 50 000 recenzija,
- dvije oznake sentimenta: `positive` i `negative`,
- uravnotežen broj pozitivnih i negativnih primjera.

Za glavne eksperimente korišten je stratificirani split:

- 40 000 recenzija za treniranje,
- 10 000 recenzija za testiranje,
- `random_state=42`,
- testni skup s 5 000 negativnih i 5 000 pozitivnih recenzija.

# 4. Metrike

Prije opisa vektorizacije i modela važno je definirati kako se uspoređuju
rezultati. Svi modeli imaju isti testni skup, pa su metrike izračunate nad istih
10 000 primjera. Za svaki model računate su sljedeće vrijednosti:

- accuracy: udio točnih predikcija,
- balanced accuracy: prosjek odziva po klasama,
- precision: koliko pozitivnih predikcija je stvarno pozitivno,
- recall: koliko stvarno pozitivnih primjera je pronađeno,
- F1: harmonijska sredina precision i recall vrijednosti,
- ROC-AUC: sposobnost rangiranja pozitivnih primjera iznad negativnih,
- PR-AUC: površina ispod precision-recall krivulje,
- MCC: Matthews correlation coefficient, korelacijska mjera pogodna za binarnu
  klasifikaciju,
- log-loss: kazna za probabilističke predikcije,
- matrica zabune.

Kod nekih modela log-loss nije prikazan. `LinearSVC`, `SGDClassifier` s hinge
loss funkcijom i `PassiveAggressiveClassifier` u ovoj konfiguraciji ne daju
kalibrirane vjerojatnosti klasa. Za njih se mogu računati tvrde predikcije i
rangirajuće vrijednosti (`decision_function`) za ROC-AUC/PR-AUC, ali log-loss
traži vjerojatnosnu distribuciju po klasama. Umjetno računanje log-loss metrike
iz samo tvrdih predikcija dovelo bi do beskonačnih ili neinformativnih kazni,
pa je u rezultatima označeno da log-loss nije dostupan za te modele. Za modele
koji daju `predict_proba`, log-loss je izračunat.

![Slika 1. Metrike za binarnu klasifikaciju sentimenta: matrica zabune definira
TN, FP, FN i TP, iz kojih se izvode accuracy, balanced accuracy, precision,
recall, F1 i MCC; ROC-AUC/PR-AUC koriste rangiranje, a log-loss traži
probabilistički izlaz.](figures/generated/evaluation_metrics.png){width=92%}

# 5. Pretprocesiranje i TF-IDF prikaz

Klasični modeli ne rade izravno nad nizovima znakova. Tekst je zato pretvoren u
rijetke numeričke vektore pomoću `TfidfVectorizer` iz biblioteke scikit-learn.
U projektu se koristi isti osnovni TF-IDF postupak za sve klasične modele, kako
bi usporedba modela bila što poštenija.

![Slika 2. Tok podataka u projektu: sirove IMDb recenzije se dijele na train i
test skup, iz trening skupa se uči TF-IDF vokabular, tekst se pretvara u
rijetku matricu značajki, a klasifikator predviđa pozitivan ili negativan
sentiment.](figures/generated/pipeline_labeled.png){width=95%}

Glavne postavke vektorizatora su:

- pretvorba teksta u mala slova,
- `strip_accents="unicode"`,
- `ngram_range=(1, 2)`, odnosno unigrami i bigrami,
- `min_df=2`, uklanjanje izraza koji se pojavljuju prerijetko,
- `max_df=0.95`, uklanjanje izraza koji se pojavljuju u gotovo svim dokumentima,
- `max_features=50000`,
- `sublinear_tf=True`.

TF-IDF ima dvije komponente. TF dio mjeri koliko se često izraz pojavljuje u
dokumentu. IDF dio smanjuje težinu izraza koji se pojavljuju u mnogo dokumenata.
Ako se neka riječ pojavljuje gotovo svugdje, ona vjerojatno nije jako korisna
za razlikovanje pozitivnih i negativnih recenzija. S druge strane, riječi ili
fraze koje su česte u jednoj recenziji, a rijetke u cijelom skupu, dobivaju
veću težinu.

Važno je da se TF-IDF vokabular uči samo na trening skupu. Testni skup se zatim
transformira pomoću već naučenog vokabulara. Time se izbjegava curenje
informacija iz testnog skupa u postupak treniranja.

![Slika 3. Intuitivni prikaz TF-IDF postupka: vokabular se uči samo na trening
recenzijama, riječi i bigrami postaju značajke, TF mjeri lokalnu važnost, IDF
smanjuje utjecaj preopćih izraza, a rezultat je rijedak vektor za
klasifikator. Slika naglašava i da TF-IDF nije BPE tokenizacija, nego fiksni
vokabular riječi i n-grama.](figures/generated/tfidf_intuitive_generated.png){width=95%}

# 6. Modeli

![Slika 4. Pregled deset propisanih klasifikatora: Bayesovi modeli uče
vjerojatnosti riječi, linearni modeli uče težine i margine, tree ensemble modeli
glasaju preko stabala, a boosting modeli dodaju stabla koja ispravljaju
prethodne pogreške.](figures/generated/ten_classifiers_overview_generated.png){width=95%}

Ovo poglavlje je strukturirano kao usporedni "model atlas": prvo se zadržava
isti TF-IDF prikaz, a zatim se mijenja algoritam učenja. Takva struktura slijedi
uobičajen način prikaza usporednih tekstualnih klasifikacijskih eksperimenata:
opis zadatka i vektorizacije, zatim modeli po skupinama, pa zajedničke metrike i
rezultati.

## 6.1 Naivni Bayesovi modeli

### 6.1.1 MultinomialNB

![Slika 5. `MultinomialNB`: model uči vjerojatnosti riječi po klasi, koristi
alpha smoothing i predviđa sentiment zbrajanjem log-vjerojatnosti za riječi iz
TF-IDF prikaza.](figures/generated/classifier_cards/multinomial_nb.png){width=95%}

`MultinomialNB` i `ComplementNB` su varijante naivnog Bayesovog klasifikatora
koje su često pogodne za tekst. Pretpostavljaju uvjetnu nezavisnost značajki
unutar klase. Ta pretpostavka nije doslovno točna za prirodni jezik, ali modeli
su brzi, stabilni i često vrlo jaki kao početni tekstualni baseline.

`MultinomialNB` procjenjuje vjerojatnost značajki po klasi i zatim za novu
recenziju zbraja log-vjerojatnosti riječi ili bigrama koje se pojavljuju u
TF-IDF vektoru. Osnovni model postigao je točnost `0.8841` i F1 `0.8846`.
Rezultat je vrlo dobar s obzirom na jednostavnost modela. Nedostatak je to što
model ne uči odnose između značajki: riječi i fraze se tretiraju kao uvjetno
nezavisni signali.

Početni NB-specific tuned eksperiment koristio je `RandomizedSearchCV` s 10
iteracija i 3 folda. Njegova testna točnost je `0.8854`, što je malo
poboljšanje, ali potvrđuje da TF-IDF parametri i `alpha` smoothing utječu na
rezultat.

### 6.1.2 ComplementNB

![Slika 6. `ComplementNB`: model procjenjuje značajke iz komplementa klase, što
je često korisno kod neuravnoteženih tekstualnih skupova; na uravnoteženom IMDb
splitu dao je isti rezultat kao `MultinomialNB`.](figures/generated/classifier_cards/complement_nb.png){width=95%}

`ComplementNB` koristi informacije iz komplementa klase i često je robusniji kod
neuravnoteženih tekstualnih skupova. U ovom projektu je skup gotovo savršeno
uravnotežen, pa su rezultati praktički identični kao kod osnovnog
`MultinomialNB`: točnost `0.8841` i F1 `0.8846`.

Ovaj rezultat ne znači da je `ComplementNB` loš model, nego da dataset ne
naglašava njegovu glavnu prednost. Oba Bayesova modela zato služe kao brzi,
interpretabilni baseline: hvataju dosta sentiment signala iz samih riječi, ali
ne uče diskriminativnu granicu jednako dobro kao linearni modeli.

## 6.2 Linearni modeli

`LogisticRegression`, `LinearSVC`, `SGDClassifier` i
`PassiveAggressiveClassifier` su linearni modeli. Kod TF-IDF značajki, broj
dimenzija je velik, ali je matrica vrlo rijetka. Linearni modeli su zato
prirodan izbor jer se mogu učinkovito trenirati nad velikim rijetkim vektorima.

### 6.2.1 LogisticRegression

![Slika 7. `LogisticRegression`: diskriminativni linearni model uči težine
TF-IDF značajki, pretvara linearni score u vjerojatnost sigmoidnom funkcijom i
može prirodno dati log-loss.](figures/generated/classifier_cards/logistic_regression.png){width=95%}

`LogisticRegression` uči linearnu granicu odluke i daje probabilističke izlaze.
U projektu je postigla točnost `0.9095` i F1 `0.9101`, što je velik skok u
odnosu na Bayesove modele. Razlog je diskriminativni cilj: model izravno uči
težine značajki koje razdvajaju pozitivnu i negativnu klasu.

Najvažnija praktična prednost je `predict_proba`. Zato logistička regresija
daje log-loss, ROC-AUC i PR-AUC na standardan način. Ako bi sustav trebao
vraćati pouzdanu procjenu sigurnosti, logistička regresija bi bila jedan od
najboljih jednostavnih kandidata.

### 6.2.2 LinearSVC

![Slika 8. `LinearSVC`: linearni SVM traži granicu odluke s velikom marginom u
rijetkom TF-IDF prostoru, optimira hinge loss i daje `decision_function` umjesto
kalibriranih vjerojatnosti.](figures/generated/classifier_cards/linear_svc.png){width=95%}

`LinearSVC` optimira SVM cilj s marginom. To je najbolji obavezni klasični model
u projektu: točnost `0.9150`, F1 `0.9152`, ROC-AUC `0.9720` i MCC `0.8300`.
SVM pristup je posebno dobar za visokodimenzionalne rijetke prostore jer uči
linearnu granicu koja ne samo da razdvaja klase, nego ih razdvaja s marginom.

Nedostatak konfiguracije korištenje u projektu je izostanak kalibriranih
vjerojatnosti. `LinearSVC` daje `decision_function`, što je dovoljno za ROC-AUC
i PR-AUC, ali ne i za standardni log-loss. Da je cilj produkcijski sustav s
pouzdanim vjerojatnostima, mogla bi se dodati kalibracija, ali za propisanu
usporedbu ovaj rezultat je najbolji klasični baseline.

### 6.2.3 SGDClassifier

![Slika 9. `SGDClassifier`: linearni model treniran stohastičkim gradijentnim
postupkom; u projektu koristi hinge loss pa se ponaša slično linearnom SVM-u,
ali s iterativnim stochastic update koracima.](figures/generated/classifier_cards/sgd_classifier.png){width=95%}

`SGDClassifier` omogućuje treniranje linearnog modela stohastičkim gradijentnim
postupkom. U ovom eksperimentu korišten je hinge loss, što ga čini sličnim
linearnom SVM-u. Postigao je točnost `0.9111` i F1 `0.9117`, vrlo blizu
`LinearSVC` rezultatu.

Prednost `SGDClassifier` modela je skalabilnost. Kada bi skup imao milijune
recenzija, stohastičko treniranje moglo bi biti praktičnije od drugih postupaka.
U ovom projektu skup je dovoljno malen da `LinearSVC` radi bez problema, ali
SGD rezultat potvrđuje da problem dobro odgovara linearnom margin-based
pristupu.

### 6.2.4 PassiveAggressiveClassifier

![Slika 10. `PassiveAggressiveClassifier`: online margin-based linearni model
ostaje pasivan kada je primjer dobro klasificiran, a agresivno ažurira težine
kada primjer krši marginu.](figures/generated/classifier_cards/passive_aggressive.png){width=95%}

`PassiveAggressiveClassifier` radi online ažuriranja i mijenja model uglavnom
kad je trenutna klasifikacija pogrešna ili nedovoljno sigurna. Postigao je
točnost `0.9062`. Njegov naziv opisuje postupak učenja: model ostaje "passive"
kada je predikcija dobra, a "aggressive" kada treba ispraviti grešku.

Na statičnom IMDb skupu rezultat je malo slabiji od `LinearSVC` i
`SGDClassifier`, ali i dalje prelazi `0.90` točnosti. Matrica zabune je gotovo
uravnotežena, s 468 lažno pozitivnih i 470 lažno negativnih grešaka, pa model
nema izrazitu pristranost prema jednoj klasi.

## 6.3 Stablima temeljeni ensemble modeli

`RandomForestClassifier` i `ExtraTreesClassifier` grade ansamble stabala
odluke. Takvi modeli su vrlo korisni na tabličnim podacima, ali na rijetkim
TF-IDF vektorima često nisu najbolji izbor. Imaju mnogo potencijalnih splitova
i mogu biti sporiji ili manje učinkoviti od linearnih modela.

### 6.3.1 RandomForestClassifier

![Slika 11. `RandomForestClassifier`: više stabala trenira se na bootstrap
uzorcima, svako stablo glasa, a konačna klasa dolazi iz većinskog glasanja ili
prosjeka vjerojatnosti.](figures/generated/classifier_cards/random_forest.png){width=95%}

`RandomForestClassifier` je postigao točnost `0.8633`, slabije od Bayesovih i
linearnih modela. To je korisna lekcija: model koji je jak na tabličnim podacima
ne mora biti najbolji za rijetke tekstualne značajke. Stabla moraju birati
splitove po pojedinim značajkama, a TF-IDF prostor ima mnogo rijetkih stupaca.

Signal sentimenta je raspršen preko velikog broja riječi i fraza, pa linearni
model koji koristi sve značajke odjednom bolje odgovara problemu.

### 6.3.2 ExtraTreesClassifier

![Slika 12. `ExtraTreesClassifier`: ansambl ekstremno randomiziranih stabala
koristi dodatnu slučajnost u odabiru značajki i pragova splitova, što ga
razlikuje od random foresta.](figures/generated/classifier_cards/extra_trees.png){width=95%}

`ExtraTreesClassifier` je postigao točnost `0.8749`, bolje od random foresta,
ali i dalje slabije od linearnih modela. Veća randomizacija u splitovima i
ansamblu ovdje pomaže, no ne mijenja osnovnu činjenicu da stabla nisu prirodan
prvi izbor za TF-IDF tekst.

U usporedbi je ipak koristan jer pokazuje razliku između dva slična bagging
pristupa. Extra Trees je bio bolji od Random Forest modela, ali oba ostaju
ispod `MultinomialNB`, `LogisticRegression`, `SGDClassifier` i `LinearSVC`
rezultata.

### 6.3.3 XGBoostClassifier

![Slika 13. `XGBoostClassifier`: gradient boosting gradi stabla sekvencijalno i
svako novo stablo pokušava ispraviti preostale pogreške; na ovom rijetkom
TF-IDF splitu model je imao najviše lažno pozitivnih pogrešaka među obaveznim
modelima.](figures/generated/classifier_cards/xgboost_classifier.png){width=95%}

`XGBoostClassifier` i `LightGBMClassifier` su gradient boosting modeli. Oni
grade niz stabala gdje svako sljedeće stablo pokušava ispraviti pogreške
prethodnih. U ovom projektu su uključeni jer su izričito navedeni u zadatku i
jer predstavljaju jaku skupinu modela za mnoge probleme strojnog učenja.

`XGBoostClassifier` je postigao točnost `0.8509`, najslabiju među obaveznim
modelima. Ovdje je problem kombinacija boosting stabala i vrlo rijetkog
tekstualnog prostora. Matrica zabune pokazuje 898 lažno pozitivnih predikcija,
što znači da je model češće označavao recenzije kao pozitivne nego što je
trebalo.

### 6.3.4 LightGBMClassifier

![Slika 14. `LightGBMClassifier`: LightGBM koristi učinkovit histogram split
search i leaf-wise rast stabala; bio je najbolji tree/boosting model u projektu,
ali nije dostigao linearne TF-IDF modele po točnosti.](figures/generated/classifier_cards/lightgbm_classifier.png){width=95%}

`LightGBMClassifier` je najbolji stablima temeljeni model u projektu. Postigao
je točnost `0.8918`, F1 `0.8925` i log-loss `0.2617`. Po točnosti ne dostiže
linearne modele, ali ima vrlo dobar probabilistički izlaz u ovom eksperimentu.

Zbog toga je LightGBM zanimljiv član soft voting ensemblea. Iako nije najbolji
po tvrdim labelama, njegove vjerojatnosti mogu biti korisne kada se prosječuju s
drugim probabilističkim modelima. To je dobar primjer zašto nije dovoljno
gledati samo accuracy; različite metrike mogu naglasiti različite kvalitete
modela.

# 7. Validacija i pretraga hiperparametara

Svi modeli su evaluirani na istom izdvojenom testnom skupu od 10 000 recenzija.
Za zahtjev `RandomizedSearchCV (CV)` proveden je zajednički tuned eksperiment za
svih deset propisanih klasičnih TF-IDF klasifikatora. Pretraga je koristila:

- `RandomizedSearchCV`,
- `n_iter=5` za svaku obitelj modela,
- `cv=3`,
- metriku za odabir `f1`,
- isti stratificirani 80/20 split kao baseline eksperimenti,
- kombinacije TF-IDF hiperparametara i model-specific hiperparametara.

To znači da je za svaku obitelj modela uzorkovano 5 kombinacija, a svaka
kombinacija je ocijenjena kroz 3 folda. Za deset klasifikatora to daje 150 CV
fitova plus završni refit najboljeg modela na cijelom trening splitu.

![Slika 15. Koncept `RandomizedSearchCV` postupka: uzorkuje se više kombinacija
hiperparametara, svaka se ocjenjuje kroz foldove unakrsne validacije, a bira se
kombinacija s najboljim srednjim rezultatom.](figures/generated/randomized_search_cv.png){width=95%}

| Model | Best CV F1 | Test ACC | Test F1 | MCC | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| MultinomialNB | 0.8800 | 0.8819 | 0.8829 | 0.7639 | 0.3044 |
| ComplementNB | 0.8800 | 0.8819 | 0.8829 | 0.7639 | 0.3044 |
| LogisticRegression | 0.8946 | 0.9029 | 0.9035 | 0.8059 | 0.2716 |
| LinearSVC | 0.9081 | 0.9168 | 0.9172 | 0.8336 | n/a |
| SGDClassifier | 0.8946 | 0.8975 | 0.8987 | 0.7952 | 0.3744 |
| PassiveAggressiveClassifier | 0.8986 | 0.9054 | 0.9054 | 0.8108 | n/a |
| RandomForestClassifier | 0.8666 | 0.8743 | 0.8765 | 0.7491 | 0.5558 |
| ExtraTreesClassifier | 0.8762 | 0.8745 | 0.8776 | 0.7499 | 0.5944 |
| XGBoostClassifier | 0.8427 | 0.8412 | 0.8475 | 0.6848 | 0.3842 |
| LightGBMClassifier | 0.8836 | 0.8922 | 0.8927 | 0.7844 | 0.2607 |

Najbolji CV rezultat i najbolji testni rezultat u ovoj široj pretrazi dao je
`LinearSVC`: srednji CV F1 `0.9081`, testna točnost `0.9168` i testni F1
`0.9172`. To potvrđuje glavnu eksperimentalnu poruku projekta: za IMDb TF-IDF
sentiment klasifikaciju dobro regulariziran linearni margin model može biti
jači od znatno složenijih tree/boosting modela.

Potpune najbolje uzorkovane konfiguracije nalaze se u tracked artefaktu
`outputs/searches/randomized_search_cv_required_n5_cv3/summary.md`. Ključni
primjeri su: `LinearSVC` je izabran s `C=0.3`, bigramima i
`max_features=30000`; `LogisticRegression` s `C=1.0` i unigramima; a
`LightGBMClassifier` s `n_estimators=200`, `num_leaves=31` i
`learning_rate=0.1`.

# 8. Rezultati klasičnih TF-IDF modela

Tablice 1 i 2 prikazuju glavne baseline rezultate za propisane klasične modele.
Uključen je i raniji tuned `MultinomialNB` redak kao dodatna Bayesova usporedba,
dok širi `RandomizedSearchCV` rezultati svih deset obitelji modela stoje u
poglavlju 7. Tablice su podijeljene zbog čitljivosti: prva prikazuje osnovne
klasifikacijske metrike, a druga metrike koje ovise o rangiranju ili
probabilističkim izlazima.

![Slika 16. Sažetak najboljih rezultata na testnom skupu od 10 000 recenzija:
najbolji klasični model je `LinearSVC`, najbolji ensemble po točnosti je hard
voting, najbolji probabilistički ensemble je soft voting, a najbolji dodatni
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

Najbolji klasični rezultat postigao je `LinearSVC`. To je očekivano za ovakav
problem: TF-IDF prostor je visokodimenzionalan i rijedak, a linearna granica
odluke je dovoljno fleksibilna za mnogo tekstualnih klasifikacijskih zadataka.
`SGDClassifier` je vrlo blizu, što dodatno potvrđuje da linearni margin-based
pristupi dobro odgovaraju ovom prikazu podataka.

Bayesovi modeli su slabiji od linearnih modela, ali su i dalje vrlo dobri s
obzirom na jednostavnost. `LightGBMClassifier` je najbolji među boosting/tree
modelima, ali ne dostiže linearne modele. `XGBoostClassifier` je ovdje najslabiji
od propisanih modela po točnosti, što pokazuje da jaki tablični modeli nisu
automatski najbolji izbor za rijetke tekstualne vektore.

## 8.1 Matrice zabune za klasične modele

U svim matricama redci su stvarne klase, a stupci predikcije. Klase su redom
`negative` i `positive`, pa su stupci u tablici:

- TN: negativna recenzija predviđena kao negativna,
- FP: negativna recenzija predviđena kao pozitivna,
- FN: pozitivna recenzija predviđena kao negativna,
- TP: pozitivna recenzija predviđena kao pozitivna.

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

`LinearSVC` ima 850 pogrešaka na 10 000 testnih primjera. Od toga je 434 lažno
pozitivnih i 416 lažno negativnih. Pogreške su relativno uravnotežene između
klasa, što je u skladu s uravnoteženim testnim skupom i sličnim vrijednostima
precision i recall metrika.

![Slika 17. Usporedba broja pogrešaka za najbolji klasični model, najbolji
ensemble po točnosti i najbolji dodatni transformer. Slika pokazuje da
transformer smanjuje ukupni broj pogrešaka, ali ostaje označen kao dodatno
proširenje.](figures/generated/error_counts.png){width=95%}

# 9. Završni ensemble modeli

Zadatak traži završni `VotingClassifier` ili `StackingClassifier`. U projektu
su implementirane tri ensemble varijante:

- soft `VotingClassifier`,
- hard `VotingClassifier`,
- prefit `StackingClassifier`.

Soft voting kombinira vjerojatnosti modela koji ih mogu dati. Prednost mu je
što zadržava probabilistički izlaz, pa je moguće računati log-loss, ROC-AUC i
PR-AUC na standardan način. Hard voting kombinira tvrde glasove i bira klasu s
većinom glasova. On je malo bolji po točnosti u ovom eksperimentu, ali nema
standardni probabilistički izlaz. Stacking uči dodatni meta-model nad izlazima
baznih modela.

![Slika 18. Usporedba ensemble arhitektura: soft voting prosječuje
vjerojatnosti, hard voting koristi većinsko glasanje, a prefit stacking koristi
spremljene bazne modele i cached meta-značajke za meta-model.](figures/generated/ensemble_architectures_project.png){width=95%}

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

Kao završni model u smislu zahtjeva kolegija može se uzeti soft
`VotingClassifier`, jer ispunjava ensemble zahtjev i daje sve probabilističke
metrike. Hard voting je također implementiran i ima nešto veću točnost, ali ne
daje log-loss bez dodatne kalibracije. Stacking model nije nadmašio voting ni
najbolji pojedinačni linearni model.

Važno je da ensemble nije automatski najbolji model. U ovom projektu najbolji
pojedinačni klasični model (`LinearSVC`) nadmašuje sve klasične ensemble
varijante.

# 10. Dodatno istraživanje: neuronski i transformerski modeli

Nakon obaveznog TF-IDF dijela isproban je dodatni smjer: mali transformerski
model od nule i fino ugađanje unaprijed naučenih modela. U izvještaju je
uključen zato što pokazuje granicu klasičnih metoda i zato što je analiza
sentimenta danas često rješavana unaprijed naučenim jezičnim modelima.
Eksperimenti su vođeni kao mala istraživačka iteracija: prvo se provjerava
može li vrlo mali transformer sam naučiti jezik zadatka, zatim se mijenjaju
vokabular, pooling, scheduler i duljina konteksta, a na kraju se isti problem
uspoređuje s predtreniranim modelima.

## 10.1 Transformer treniran od nule

![Slika 19. Mali transformer treniran od nule: BPE vokabular od 10 000 tokena,
embedding dimenzija 24, četiri transformer encoder sloja, mean pooling i
klasifikacijska glava za pozitivan ili negativan sentiment.](figures/generated/tiny_transformer_from_scratch.png){width=95%}

Prvi cilj bio je namjerno strog: transformer ne smije koristiti predtreniranu
bazu, nego mora naučiti reprezentaciju iz samih IMDb oznaka. Početna MLX
varijanta imala je BPE vokabular od samo 256 tokena, `max_length=128`,
`d_model=16`, četiri encoder sloja i oko 13 000 parametara. Iako se takav model
brzo trenirao, testna točnost bila je samo `0.6385`. Taj rezultat je pokazao da
je model previše informacijski ograničen: s vrlo malim vokabularom i kratkim
kontekstom veći dio recenzije se ili jako fragmentira ili odsiječe.

Zato je napravljena druga serija eksperimenata. BPE tokenizer povećan je na
10 000 tokena i spremljen kao cache, kako se ne bi ponovno učio pri svakom
pokretanju. Uveden je mean pooling preko tokena, dropout `0.1`, AdamW optimizator
i warmup-cosine raspored stope učenja. Najbolji model treniran od nule imao je
`d_model=24`, četiri encoder sloja, četiri attention glave, `ff_dim=48`,
`max_length=512` i `263 306` parametara. Trening je izveden u MLX-u s
mikro-batchevima od 4 primjera i efektivnim batchem 16. Najbolja validacijska
točnost bila je `0.8935`, a testna točnost `0.8943`.

Ovaj rezultat je znatno bolji od početne sitne varijante, ali i dalje ne
nadmašuje `LinearSVC` s TF-IDF značajkama. To je bitan zaključak: arhitektura
transformera sama po sebi ne jamči bolji rezultat ako model nema dovoljno
predtreniranja ili dodatnog tekstualnog signala. Na udaljenoj RTX 3090 instanci
isproban je i PyTorch/CUDA ekvivalent radi provjere brzine i duljeg konteksta.
Varijanta s kontekstom 512 trenirala je oko dvije minute i postigla točnost
`0.8733`, a varijanta s kontekstom 1024 oko 3.7 minuta i točnost `0.8840`.
Dulji kontekst je pomogao, ali nije zatvorio razliku prema najboljim linearnim
TF-IDF modelima.

## 10.2 Fine-tuning predtreniranih transformera

Zatim su fino ugođena dva unaprijed naučena modela:

- `distilbert-base-uncased`,
- `microsoft/deberta-v3-small`.

![Slika 20. Fine-tuning predtreniranog transformera: model prvo uči opće jezične
reprezentacije na velikim korpusima, a zatim se prilagođava IMDb oznakama
sentimenta pomoću klasifikacijske glave.](figures/generated/pretrained_transformer_finetuning.png){width=95%}

Treniranje je izvedeno na udaljenoj Vast.ai RTX 3090 instanci.

Kod fine-tuninga se više ne uči jezična reprezentacija od nule. Predtrenirani
model već ima naučene obrasce engleskog jezika, sintakse, konteksta i
semantičkih sličnosti iz velikih korpusa. IMDb trening skup tada služi za
prilagodbu klasifikacijske glave i unutarnjih reprezentacija na konkretan
zadatak pozitivnog ili negativnog sentimenta. Korišten je AdamW s
`learning_rate=2e-5`, `weight_decay=0.01`, warmup omjerom `0.06` i mixed
precision treniranjem na CUDA uređaju.

U sljedećoj tablici "Start" označava je li model krenuo od random
inicijalizacije ili od predtrenirane jezične baze s novom klasifikacijskom
glavom. `Val@1` je validacijska točnost nakon prve epohe fine-tuninga.

| Model | Start | Val@1 | Best val | Test ACC | Greške |
| --- | --- | ---: | ---: | ---: | ---: |
| Tiny | scratch | 0.8613 | 0.8935 | 0.8943 | 1057 |
| DistilBERT | pretrained + nova glava | 0.9220 | 0.9340 | 0.9369 | 631 |
| DeBERTa | pretrained + nova glava | 0.9493 | 0.9565 | 0.9564 | 436 |

| Model | Prec. | Rec. | F1 | ROC-AUC | PR-AUC | Log-loss | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 0.8796 | 0.9136 | 0.8963 | 0.9603 | 0.9592 | 0.2789 | 0.7892 |
| DistilBERT | 0.9296 | 0.9454 | 0.9374 | 0.9840 | 0.9831 | 0.1888 | 0.8739 |
| DeBERTa | 0.9521 | 0.9612 | 0.9566 | 0.9895 | 0.9886 | 0.1561 | 0.9128 |

Najbolji ukupni model je `microsoft/deberta-v3-small`. On ima 436 pogrešaka na
10 000 testnih primjera, dok `LinearSVC` ima 850 pogrešaka. Razlika je velika,
ali dolazi iz unaprijed naučenog jezičnog znanja, a ne samo iz arhitekture. To
je glavna pouka dodatnog eksperimenta: na ograničenom označenom skupu, model s
predtreniranim razumijevanjem jezika može puno bolje generalizirati nego model
učen od nule.

# 11. Rasprava

Rezultati pokazuju nekoliko jasnih obrazaca.

Prvo, TF-IDF i linearni modeli su vrlo jaki za ovaj zadatak. `LinearSVC`,
`SGDClassifier` i `LogisticRegression` svi prelaze točnost od `0.90`. To je
dobar rezultat s obzirom na jednostavnost prikaza i činjenicu da se ne koristi
nikakvo duboko jezično predznanje.

Drugo, naivni Bayesovi modeli su korisni kao brz baseline. Njihova točnost oko
`0.884` nije najbolja, ali je dovoljno visoka da pokaže kako se veliki dio
signala nalazi u samim riječima i frazama. Tuniranje `MultinomialNB` modela
donijelo je malo poboljšanje, ali nije promijenilo ukupni poredak modela.

Treće, modeli temeljeni na stablima nisu optimalni za ovaj TF-IDF prikaz.
`LightGBMClassifier` je najbolji iz te skupine, ali i dalje zaostaje za
linearnim modelima. Razlog je vjerojatno priroda značajki: TF-IDF matrica je
vrlo rijetka i visokodimenzionalna, a linearni modeli mogu izravno iskoristiti
takvu strukturu.

Četvrto, ensemble modeli nisu automatski bolji od najboljeg pojedinačnog
modela. Voting i stacking modeli stabilno rade, ali ne nadmašuju `LinearSVC`.
To je korisna praktična lekcija: ensemble je tehnika koja može pomoći kada
bazni modeli imaju različite pogreške, ali nije jamstvo boljeg rezultata.

Peto, dodatni transformerski eksperimenti pokazuju veliku razliku između modela
učenog od nule i modela s predtreniranjem. Mali transformer od nule ne uspijeva
nadmašiti TF-IDF baseline, dok DeBERTa-v3-small znatno nadmašuje sve klasične
modele. Time se vidi vrijednost transfer learninga u obradi prirodnog jezika.

# 12. Implementacijska organizacija

Projekt nije rađen kao notebook-first projekt, nego kao Python workflow. To je
namjerno, jer je lakše pratiti svaki model kao zaseban eksperiment i ponovno
pokretati pojedine dijelove. Svaki klasifikator ima vlastitu mapu u
`classifiers/`, obično s `run.py` skriptom i `README.md` datotekom.

Zajedničke funkcije nalaze se u `src/sentiment/`:

- `data.py` učitava Kaggle CSV i priprema splitove,
- `metrics.py` računa metrike i matricu zabune,
- `artifacts.py` sprema JSON i tekstualne artefakte,
- `paths.py` definira standardne putanje.

Takva organizacija smanjuje ponavljanje. Pojedinačni modeli mogu se razlikovati
u hiperparametrima i načinu treniranja, ali dijele isti način učitavanja
podataka, istu evaluaciju i isti oblik izlaznih artefakata. To je važno za
poštenu usporedbu: ako bi svaki model imao drugačiji split ili drugačije
računanje metrika, rezultati ne bi bili izravno usporedivi.

Za svaki eksperiment sprema se `run_config.json`. Ta datoteka je posebno važna
jer dokumentira što je točno pokrenuto: parametre TF-IDF vektorizatora, broj
redaka, split, model i dostupnost probabilističkih izlaza. Brojčani rezultati u
izvještaju su preuzeti iz `metrics.json` i `confusion_matrix.json` artefakata.

# 13. Analiza grešaka i mogući sljedeći koraci

Ovaj projekt ne uključuje duboku kvalitativnu analizu pojedinačnih pogrešno
klasificiranih recenzija, ali se iz matrica zabune mogu izvući korisni
zaključci. Kod najboljeg klasičnog modela, `LinearSVC`, broj lažno pozitivnih
i lažno negativnih primjera gotovo je jednak. To znači da model nema očit
pomak prema jednoj klasi. Kod `XGBoostClassifier` modela vidi se drugačija
slika: broj lažno pozitivnih predikcija je znatno veći, pa model češće
prepoznaje recenzije kao pozitivne.

U stvarnom sustavu analiza grešaka bila bi sljedeći korak. Posebno bi bilo
korisno pregledati:

- sarkastične recenzije,
- recenzije s miješanim sentimentom,
- recenzije koje pozitivno govore o jednom aspektu filma, ali negativno o
  ukupnom dojmu,
- vrlo kratke recenzije,
- recenzije s riječima koje imaju drugačije značenje ovisno o kontekstu.

TF-IDF modeli teško razumiju ironiju, negaciju preko dužeg konteksta i
semantičke odnose. Bigrami pomažu kod kratkih fraza poput "not good", ali ne
rješavaju cijeli problem. To objašnjava zašto predtrenirani transformer može
biti bolji: on dolazi s već naučenim jezičnim reprezentacijama i može bolje
iskoristiti kontekst.

Mogući nastavci projekta bili bi:

- dodatna kalibracija `LinearSVC` modela,
- tuning `LogisticRegression` i `LightGBMClassifier` modela,
- optimizacija težina soft voting ensemblea,
- evaluacija na Stanford ACL IMDB splitu kao dodatnom vanjskom testu,
- interpretacija značajki za linearne modele,
- analiza pogrešno klasificiranih primjera.

# 14. Reproducibilnost

Projekt je organiziran kao Python repozitorij. Glavni kod je u mapama:

- `src/sentiment/` za zajedničke funkcije učitavanja podataka, metrika i
  putanja,
- `classifiers/` za pojedinačne modele,
- `outputs/` za generirane rezultate,
- `reports/` za izvještajne materijale.

Primjer pokretanja jednog obaveznog modela:

```bash
PYTHONPATH=. python -m classifiers.linear_svc.run
```

Primjer pokretanja široke `RandomizedSearchCV` provjere za svih deset
klasifikatora:

```bash
PYTHONPATH=. python -m classifiers.randomized_search_cv.run \
  --models all \
  --n-iter 5 \
  --cv 3 \
  --scoring f1 \
  --output-dir outputs/searches/randomized_search_cv_required_n5_cv3
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

Transformerski eksperimenti su računalno zahtjevniji i nisu potrebni za osnovno
pokretanje projekta. Pokretani su na GPU instanci, a rezultati su prepisani u
izvještaj.

# 15. Ograničenja

Glavno ograničenje projekta je to što je primarna evaluacija napravljena na
jednom splitu Kaggle IMDb skupa. Iako je split stratificiran i velik, dodatna
evaluacija na potpuno odvojenom izvoru mogla bi dati još jaču procjenu
generalizacije.

Drugo ograničenje je dubina hiperparametarske pretrage. `RandomizedSearchCV` je
proveden i dokumentiran za svih deset propisanih klasičnih modela, ali s
namjerno ograničenim budžetom `n_iter=5` i `cv=3` po modelu. To je praktičan
kompromis: deset modela, ensemble varijante i dodatni transformerski
eksperimenti već daju velik usporedni prostor. Za maksimalno optimiranje
klasičnih modela bilo bi korisno povećati `n_iter`, proširiti distribucije
parametara, dodati kalibraciju za margin modele i pretražiti ensemble težine.

Treće ograničenje je log-loss kod modela bez probabilističkog izlaza. Umjesto
da se ta metrika umjetno računa iz tvrdih predikcija, u izvještaju je označeno
da nije dostupna. Alternativa bi bila kalibracija modela pomoću
`CalibratedClassifierCV`, ali to bi promijenilo modele i dodalo još jedan sloj
treniranja.

# 16. Zaključak

Projekt ispunjava osnovne zahtjeve teme 7: koristi IMDb recenzije, TF-IDF
prikaz, deset propisanih klasifikatora, `RandomizedSearchCV` s unakrsnom
validacijom, standardne metrike i završne Voting/Stacking ensemble modele.

Najbolji klasični model je `LinearSVC` s točnosti `0.9150`. Najbolji obavezni
ensemble je hard `VotingClassifier` po točnosti, dok je soft `VotingClassifier`
prikladniji kao završni probabilistički ensemble jer daje log-loss i
vjerojatnosti. Ensemble modeli nisu nadmašili `LinearSVC`, što pokazuje da je
za TF-IDF tekstualne značajke dobro odabran linearni model vrlo jak baseline.

Dodatni eksperimenti s transformerima pokazuju da moderno predtreniranje može
znatno poboljšati rezultat. `DeBERTa-v3-small` postiže točnost `0.9564`, što je
znatno više od klasičnih TF-IDF modela.

# Literatura i izvori

- Materijali kolegija: `PUI - Osnovne Informacije o kolegiju.pdf`.
- Kaggle: IMDb 50K Movie Reviews.
- Stanford AI Lab: IMDb Large Movie Review Dataset.
- scikit-learn dokumentacija za `TfidfVectorizer`, linearne modele, naive Bayes
  modele, ensemble modele i metrike.
- scikit-learn example: "Classification of text documents using sparse
  features", koristan kao struktura za usporedbu više klasifikatora nad istim
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

Ovi artefakti su izvor brojčanih rezultata prikazanih u izvještaju.

# Dodatak B: Komande za ponavljanje eksperimenata

Sljedeća tablica navodi komande za glavne pokrete. Komande se izvode iz korijena
repozitorija. Ako paket nije instaliran u editable modu, koristi se
`PYTHONPATH=.` prefiks kao u primjerima.

| Eksperiment | Komanda | Glavni izlaz |
| --- | --- | --- |
| MultinomialNB baseline | `PYTHONPATH=. python -m classifiers.multinomial_nb.run` | `outputs/baselines/multinomial_nb` |
| MultinomialNB tuned | `PYTHONPATH=. python -m classifiers.multinomial_nb.run --tune --n-iter 10 --cv 3 --output-dir outputs/baselines/multinomial_nb_tuned_n10_cv3` | `outputs/baselines/multinomial_nb_tuned_n10_cv3` |
| RandomizedSearchCV svih 10 modela | `PYTHONPATH=. python -m classifiers.randomized_search_cv.run --models all --n-iter 5 --cv 3 --scoring f1 --output-dir outputs/searches/randomized_search_cv_required_n5_cv3` | `outputs/searches/randomized_search_cv_required_n5_cv3` |
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

Svaki od baseline i ensemble izlaznih direktorija sadrži barem `metrics.json`,
`confusion_matrix.json`, `classification_report.txt` i `run_config.json`.
Direktorij `outputs/searches/randomized_search_cv_required_n5_cv3/` dodatno
sadrži `summary.csv`, `summary.md` i po jedan `search_results.csv` za svaki od
deset tuned modela. Većina `outputs/` direktorija nije namijenjena predaji kao
veliki generirani artefakt, ali je ovaj compact CV direktorij sačuvan kao dokaz
za zahtjev `RandomizedSearchCV`. Završni rezultati su dodatno prepisani u
`reports/final_report/results_table.csv` i
`reports/final_report/results_tables.md`.

# Dodatak C: Pregled konfiguracija klasičnih modela

Svi klasični baseline modeli koriste isti osnovni dataset, split i TF-IDF
prikaz. Osnovni split je 40 000 trening recenzija i 10 000 testnih recenzija, s
`random_state=42`. Dodatni `RandomizedSearchCV` eksperiment u poglavlju 7
pretražuje TF-IDF i model-specific parametre za svih deset obitelji modela.

| Model | Glavni parametri modela | Probabilistički izlaz | Napomena |
| --- | --- | --- | --- |
| MultinomialNB | `alpha=1.0` | da | brzi tekstualni baseline |
| MultinomialNB tuned | `alpha=0.5`, tuned TF-IDF | da | `RandomizedSearchCV`, `n_iter=10`, `cv=3`, najbolji CV F1 `0.8850` |
| ComplementNB | `alpha=1.0` | da | isti rezultat kao MultinomialNB na uravnoteženom splitu |
| LogisticRegression | `C=1.0`, `solver=liblinear`, `max_iter=1000` | da | jak linearni probabilistički baseline |
| LinearSVC | `C=1.0`, `dual=auto`, `max_iter=5000` | ne | najbolji klasični model po točnosti/F1 |
| SGDClassifier | `loss=hinge`, `alpha=0.0001`, `max_iter=1000` | ne | SVM-sličan linearni model treniran SGD-om |
| PassiveAggressiveClassifier | `C=1.0`, `max_iter=1000` | ne | online margin-based model |
| RandomForestClassifier | `n_estimators=100`, `max_features=sqrt` | da | slabiji na rijetkim TF-IDF značajkama |
| ExtraTreesClassifier | `n_estimators=100`, `max_features=sqrt` | da | bolji od random foresta, ali slabiji od linearnih modela |
| XGBoostClassifier | `n_estimators=200`, `max_depth=4`, `tree_method=hist` | da | boosting model s izraženim FP greškama u ovom splitu |
| LightGBMClassifier | `n_estimators=200`, `num_leaves=31`, `learning_rate=0.1` | da | najbolji tree/boosting model i najniži klasični log-loss |

Osnovni TF-IDF parametri su `ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`,
`max_features=50000`, `strip_accents="unicode"` i `sublinear_tf=True`.
RandomizedSearchCV pretraga dodatno uzorkuje `ngram_range`, `min_df`, `max_df`,
`max_features` i `sublinear_tf`, uz parametre samih klasifikatora. Vokabular se
uvijek uči samo na trening podacima.

# Dodatak D: Detalji ensemble modela

Soft `VotingClassifier` koristi tri probabilistička člana:
`LogisticRegression`, `MultinomialNB` i `LightGBMClassifier`. Ovaj izbor je
praktičan jer sva tri člana daju `predict_proba`, pa se mogu računati
standardni log-loss, ROC-AUC i PR-AUC. Njegov log-loss `0.2654` bolji je od
log-loss vrijednosti same logističke regresije (`0.2707`), iako mu točnost nije
bolja od `LinearSVC` modela.

Hard `VotingClassifier` uključuje sve glavne klasične modele, uključujući i
tuned `MultinomialNB` varijantu. Ima najbolju ensemble točnost `0.9107`, ali
nema standardni probabilistički izlaz. U zasebnom README-u zabilježene su i
vote-fraction vrijednosti kao gruba ordinalna mjera, no one se u glavnoj
tablici ne tretiraju kao standardni log-loss jer nisu kalibrirane
vjerojatnosti.

Prefit stacking koristi spremljene bazne modele i gradi meta-značajke iz
njihovih izlaza. Za modele s `predict_proba` koristi probabilističke izlaze, a
za `LinearSVC`, `SGDClassifier` i `PassiveAggressiveClassifier` koristi
`decision_function`. Meta-model je logistička regresija. Ova implementacija je
praktična jer izbjegava ponovno treniranje svih baznih modela, ali je u
izvještaju jasno označena kao prefit/cached stacking, a ne kao skuplja
out-of-fold stacking procedura.

# Dodatak E: Detalji transformerskog proširenja

Transformerski dio ima dvije različite svrhe. Mali transformer od nule služi
kao istraživanje koliko se može naučiti bez predtreniranja. Pretrained
transformeri služe kao usporedba s modernijim NLP pristupom koji koristi
predznanje naučeno na mnogo većim korpusima.

| Model | Parametri | Max length | Trening | Najbolja val točnost | Test točnost | Runtime |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Tiny MLX početni pokus | oko 13k | 128 | MLX, 256 BPE tokena | oko `0.646` | `0.6385` | oko 70 s |
| Tiny MLX najbolji od nule | 263k | 512 | MLX, 10k BPE, mean pooling, mikro-batch 4 | `0.8935` | `0.8943` | oko 17.1 min |
| Tiny Torch CUDA kontekst 1024 | 263k | 1024 | RTX 3090, fp16, batch 16 | `0.8742` | `0.8840` | oko 3.7 min |
| DistilBERT | 66.96M | 512 | RTX 3090, 3 epohe, batch 32 | `0.9340` | `0.9369` | oko 10.8 min |
| DeBERTa-v3-small | 141.90M | 512 | RTX 3090, 3 epohe, batch 16 | `0.9565` | `0.9564` | oko 25.9 min |

Za `DistilBERT` i `DeBERTa-v3-small` korišten je AdamW, `learning_rate=2e-5`,
`weight_decay=0.01`, warmup omjer `0.06` i mixed precision na CUDA uređaju.
`DistilBERT` je koristio batch 32, a `DeBERTa-v3-small` batch 16. Oba modela
koriste 36 000 primjera za treniranje, 4 000 za validaciju i 10 000 za test.

# Dodatak F: Kontrolna lista zahtjeva

Ova kontrolna lista služi kao završna provjera usklađenosti s temom 7.

| Zahtjev | Status u projektu | Dokaz |
| --- | --- | --- |
| Analiza sentimenta recenzija | ispunjeno | Kaggle IMDb recenzije, binarne oznake `positive`/`negative` |
| TF-IDF prikaz | ispunjeno | `TfidfVectorizer` u klasičnim pipelineovima |
| Najmanje 10 klasifikatora | ispunjeno | 10 propisanih obitelji modela |
| RandomizedSearchCV | ispunjeno | svih 10 propisanih modela, `n_iter=5`, `cv=3` |
| Cross-validation | ispunjeno | 3-fold CV u široj RandomizedSearchCV pretrazi |
| ACC, BACC, precision, recall, F1 | ispunjeno | tablice rezultata u poglavlju 8 |
| ROC-AUC i PR-AUC | ispunjeno | računato iz vjerojatnosti ili `decision_function` |
| MCC | ispunjeno | tablice rezultata u poglavlju 8 |
| Log-loss | ispunjeno gdje je matematički dostupan | n/a objašnjen za modele bez kalibriranih vjerojatnosti |
| Matrice zabune | ispunjeno | poglavlja 8, 9 i 10 |
| Voting ili Stacking finalni model | ispunjeno | soft/hard voting i prefit stacking |
| Izvještaj u prihvatljivom formatu | ispunjeno | `reports/final_report/analiza_sentimenta_recenzija.docx` |
| Prezentacija u prihvatljivom formatu | ispunjeno | `presentation/analiza_sentimenta_recenzija.pptx` |
| Upute za pokretanje | ispunjeno | `README.md` i poglavlje 14 |

Preostali kompromisi su namjerno navedeni u poglavlju ograničenja: modeli nisu
iscrpno optimirani, primarna evaluacija je jedan stratificirani Kaggle split, a
nekalibrirani margin-based modeli nemaju log-loss.
