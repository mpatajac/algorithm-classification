Cilj zadatka je klasifikacija teksta, odnosno za dani tekst (u ovom slučaju recenzija filma na IMDB) odrediti ima li taj tekst pozitivan ili negativan sentiment.

Dataset se nalazi na poveznici https://ai.stanford.edu/%7Eamaas/data/sentiment/

Train/Test split je već napravljen. Unutar train i test direktorija, u direktoriju pos se nalaze tekstovi s pozitivnim, a u neg s negativnim sentimentom.

Osim tog dataseta, još ti je korisna datoteka imdb.vocab, u kojoj se nalaze sve riječi iz dataseta. To će ti pomoći za indeksiranje riječi i njihovo preslikavanje u vektorski prostor.

Tvoj zadatak je sljedeći:
 - Koristeći Dataset i DataLoader klase iz torch.utils.data (vidi https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), učitaj cijeli dataset, uredi ga tako da izbaciš sve nepotrebne znakove i interpunkcije, i svaku riječ preslikaj u pripadni indeks iz imdb.vocab. Za tokenizaciju čak postoje neki libraryji, ali ovdje ni ručna tokenizacija nije previše zahtjevna.
 - Pripazi na pretvaranje numeričkih vrijednosti u tekstualne, npr. 35 -> thirty-five
 - Kad DataLoader konstruira minibatch od npr. 64 tekstova, traži da svi budu jednake duljine. Općenito u RNN modelima nizovi nisu jednake duljine pa ih je potrebno popuniti nekim trash vrijednostima. Za taj problem postoji rješenje u PyTorchu, vidi https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html ili dokumentaciju.
 - Kako bi mogli baratati tim riječima, potrebno ih je preslikati u neki vektorski prostor. Na vježbama iz SU smo koristili one-hot enkodiranje za slova, ali ovdje takav pristup nije nimalo efikasan jer imamo ~90000 riječi pa je dimenzija prostora ogromna. Zato možeš koristiti torch.nn.Embedding (embedding općenito znači ugrađivanje u neki vektorski prostor) u kojem zadaš dimenziju prostora (npr. 100), generiraju se nasumični embeddinzi za sve riječi i zajedno se uče s ostatkom modela.
 - Još jedan pristup za embeddinge je word2vec ili GloVe, ali mislim da nije potrebno za ovaj zadatak.
 - Najvažniji dio zadatka je definirati model koji će raditi kvalitetnu klasifikaciju. Jedna ideja je koristeći neku RNN arhitekturu (LSTM ili GRU) proći kroz tekst riječ po riječ i za svaku riječ i prethodni kontekst generirati skrivena stanja. U zadnjem skrivenom stanju je enkodiran cijeli niz i može se iskoristiti za klasifikaciju. Primjerice, ako je skriveno stanje dimenzije 200, možeš ga nekako pretvoriti u skalar, dobiti realnu vrijednost i nju interpretirati kako god želiš. Za loss funkciju predlažem da koristiš binary cross entropy funkciju (https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html).
 - Za korištenje LSTM/GRU morat ćeš dobro pročitati dokumentaciju (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) ili pogledati već neki implementirani model (https://github.com/pytorch/examples/tree/master/word_language_model)

Kada riješiš ovaj zadatak, jednostavno ćeš ga "prenamijeniti" za klasifikaciju programskog koda umjesto rečenica.

Ako imaš bilo kakvih pitanja, slobodno pitaj mene ili profesora. :)

---
## Nastavak zadatka

U LSTM dodaj bidirectional parametar kako bi niz "obradio" u oba smjera i vidi hoće li se rezultati poboljšati.
Za detalje vidi npr. Aggarwal, poglavlje 7.2.3, str. 283.

Nakon enkoder sloja implementiraj attention sloj (vidi <https://towardsdatascience.com/attention-in-neural-networks-e66920838742>, soft attention) ili Aggarwal poglavlje 10.2.2, str. 425.

S attention slojem želiš vidjeti koliko koje hidden stanje (ili ulazni token) utječe na završni rezultat.

Rezultat attentiona možeš npr. zbrojiti s zadnjim hidden stanjem enkodera ili ga konkatenirati i dobiti vektor duplo veće dimenzije.

---
# Novi zadatak

Nakon što poboljšaš ovaj jezični model, probaj isti model primijeniti za klasifikaciju programskog koda.

Na <https://drive.google.com/file/d/0B2i-vWnOu7MxVlJwQXN6eVNONUU/> se nalazi dataset s 104 različita algoritma, po 500 implementacija za svaki algoritam. Svi programi su napisani u C++ jeziku. Jedan pristup klasifikaciji je prevesti program u LLVM IR reprezentaciju i promatrati program kao niz LLVM IR instrukcija, isto kao što je tekst niz riječi. LLVM IR možeš dobiti naredbom

   clang -S -emit-llvm prog.cpp

Pripremljene LLVM IR reprezentacije tih programa možeš preuzeti na <https://polybox.ethz.ch/index.php/s/JOBjrfmAjOeWCyl>. Ovdje je već napravljen train/validation/test split.

Jedan problem je konstrukcija embeddinga za LLVM IR instrukcije. Naime, broj mogućih instrukcija je ogroman te ih je potrebno oblikovati na način da stanu u neki manji vokabular. Primjerice, uklanjanjem identifikatora ili nekih numeričkih vrijednosti smanjuje se broj mogućih instrukcija (iako je još uvijek ogroman).
Zato predlažem da pročitaš i iskoristš skripte <https://github.com/spcl/ncc/blob/master/rgx_utils.py> i <https://github.com/spcl/ncc/blob/master/task_utils.py> te vokabular <https://polybox.ethz.ch/index.php/s/AWKd60qR63yViH8>.

Ako imaš neku svoju ideju za obradu LLVM IR instrukcija, slobodno je probaj implementirati.

Zasad ne moraš koristiti neke naučene embeddinge za ove instrukcije, nego vidjeti kakve će rezultate dati model ako uzmeš proizvoljne ili one-hot embeddinge.

Trebaš konstruirati model sličan prethodnom, koji umjesto embeddinga za riječi prima embeddinge LLVM IR instrukcije i outputa jednu od 104 klase. Sretno!