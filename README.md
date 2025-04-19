# Privacy in dynamic graphs
## Anonymization and community detection in temporal-varying graphs
> For this project, we consider dynamic graphs. That 
> is, graphs which have a temporal component. In such graphs we might find
> vertices or even nodes which appear or disappear in a temporal basis.
> Anonymizing these graphs introduces important challenges not found in
> more classical static graphs. In this project, we aim to explore
> possible solutions for the anonymization of temporal graphs by relaying
> in well known privacy models such as k-anonymity and differential privacy.

El problema de protecció de dades està a l'ordre del dia, on és crític dintre de molts sectors, com pot ser el de salut, tecnològic, finances, etc.
Un tema preocupant són les xarxes socials o altres tipus de xarxes que generen un gran volum de dades i interaccions en cada instant,
on està la necessitat de mantenir-les protegides en tot moment. Aquestes dades es poden interpretar
com a grafs, on els nodes representen objectes o usuaris, i les arestes són les relacions entre dos nodes dintre de la xarxa.
Des del punt de vista d'una xarxa, els nodes i arestes poden anar canviant durant el temps, i és necessari tractar aquests grafs
per tal d'evitar atacs que poden portar a identificar l'identitat o informació personal dels usuaris. 

## 1. Objectius
Per tant, l'objectiu és investigar sobre els grafs que varien durant el temps, on es volen assolir els següents punts:

* Establir les definicions bàsiques d'un graf temporal i quines propietats addicionals té en comparació dels grafs estàtics.

* Aplicar diversos mètodes de privacitat a diferents conjunts de datasets, des de volums de dades fàcils de tractar, 
fins a una quantitat massiva de dades. Pels mètodes de privacitat, esquematitzar quins són i com funcionen. L'intenció 
és que els mètodes siguin els més consistents i òptims possibles, on es vol fer un estudi de quin són els millors paràmetres per cada conjunt de dades. 

* Realitzar simulacions d'atacs a la xarxa per cada mètode de protecció, i analitzar des del punt de vista del atacant 
què ha de passar per retreure informació d'un individu, tant en un instant de temps, com durant tota l'evolució temporal.

* Fer una comparativa a nivell de privacitat i utilitat dels grafs protegits i original. Una mètrica que es vol estudiar i donar importància és com es fa la detecció
de comunitats, i com aquestes evolucionen durant el temps. Per tant també es vol analitzar quines formes hi ha per detectar comunitats i fer comparacions entre elles.

* Intentar implementar una xarxa neuronal que permet predir quina serà l'evolució d'un graf en el següent instant de temps, i que compleixi els termes de privacitat dels mètodes.
 
* Provar a visualitzar a mode d'animació com es protegeixen les dades, o bé la comparació de detecció de comunitats durant el temps.

## 2. Metodologia

Com s'ha esmentat en l'apartat 1, es volen aplicar diferents mètodes de privacitat. 
Els casos d'estudi son diversos tipus de datasets trobats en [1,2]. 
La llista de datasets que es conté per defecte és la de a continuació:

<div align="center">
  
| Name | # Vertices | # Edges | Is directed | Has weight | # Snapshots | # HOUR | # DAY | # WEEK 
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Aves-sparrow | 52 | 516 | False | True | 2 | - | - | - |
| Mammalia-voles | 1480 | 4569 | False | True | 61 | - | - |  
| Insecta-ant | 152 | 194K | False | True | 41 | - | - | - |
| Enron-Employees | 151 | 50.5K | True | True | 16067 | 6440 | 867 | 161 |
| CollegeMsg | 1899 | 59.8K | True | False | 58911 | 3320 | 193 | 29 |

</div>

El llenguatge de programació que s'ha fet servir principalment ha estat Python v3.13.0, utilitzant les llibreries:

* NetworkX (v3.2.1): Per a la creació dels grafs a partir de les dades dels datasets. 

* Pandas (v2.1.4): Per fer conversió de dades en DataFrames. Ús principalment per fer agregacions de dades.

* Matplotlib (v3.8.2): Utilitzat per visualització de gràfics. Eficient per observar mètriques.

* Tensorflow (v2.17.0): Per a crear una xarxa neural, en cas de voler fer un forecasting de grafs. 

Com a planificació personal, s'ha realitzat un diagrama de Gantt, estimant en quin moment s'ha de tenir cada possible tasca del treball finalitzada.
Les tasques estan dividides segons si forma part a la part conceptual del projecte, o bé si és a nivell pràctic. 
La part teòrica del treball engloba a la recerca d'informació i la definició dels mètodes. 
En canvi, l'altre part s'enfoca en la recerca de dades, aplicar els mètodes de protecció i detecció de comunitats estudiats, i finalment fer una anàlisi de mètriques. 

## 3. Estat de l'art

Les maneres de protegir grafs estàtics són diverses, 
com per exemple fer que els nodes siguin indistinguibles entre ells a partir dels seus atributs, 
afegir soroll, estratègies de xifratge, etc. En els grafs dinàmics es troben certes semblances amb les maneres de protegir les dades,
però tenen una major complexitat ja que es conté un factor temporal. Principalment es volen implementar el K-Anonimity [3]
i el Edge-Local Differential Privacy [4] aplicats en grafs temporals. 

Similarment, en el moment de calcular la privacitat i utilitat, si volem saber la similitud entre grafs, es poden usar altres versions del \textit{Coeficient de Jaccard} o la \textit{Cosine Similarity} \textit{[5,6]}. Si es necessita implementar algorismes de detecció de comunitats com a informació d'utilitat, es té \textit{[7]}.

## 4. Definicions bàsiques - Mètodes de privacitat

* Graf dinàmic: (Definition problem -> K-An) 

* Diferències entre grafs estàtics i grafs dinàmics: Un graf dinàmic resumidament conté una variable temporal afegida comparat amb els grafs estàtics.
En què pot afectar això? Doncs s'ha de vigilar més com es protegeixen les dades, perquè un atacant pot obtenir més informació si conté actualitzacions d'una xarxa. 
Què pot fer un atacant que no pugui en un graf estàtic protegit? Pot anar recollint grafs temporals i anar veient la relació entre les dades, fins poder re-identificar els nodes i relacions reals. 

* Per LEDP: Def 1. Def 2. Def 3. Proposition 1 (DP in dynamic graphs - Julián)
    * Anàlisi: L'algorisme afegeix un soroll a cada graf que es conté de forma individual i paral·lelament, que compleix ε-Edge Local DP. Un atacant (només amb l'informació protegida) no pot treure informació dels grafs individuals, ja que aquests contenen un soroll que ell no coneix. És a dir, no pot conèixer si les relacions dels nodes són reals o no. Si es tenen tots els grafs protegits, de forma global només es pot arribar a saber quina era aproximadament la densitat mitjana dels grafs originals, ja que aquest algorisme permet preservar-la (afegir/treure soroll d'una forma anivellada). Ara bé, si es tenen els grafs originals, es pot arribar a veure a partir d'aquests quines són les relacions reals dels nodes, i es podria inferir quin soroll s'ha afegit de forma específica en cada graf. 


## 5. Referències

[1] J. Leskovec, Stanford Network Analysis Project (SNAP). Disponible en: https://snap.stanford.edu/index.html. [Darrer accés: 26-feb-2025].

[2] Ryan A. Rossi i Nesreen K. Ahmed, The Network Data Repository with Interactive Graph Analytics and Visualization, 2015. 
Disponible en: https://networkrepository.com/dynamic.php. [Darrer accés: 26-feb-2025].

[3] L. Rossi, M. Musolesi i A. Torsello, "On the k-Anonymization of Time-Varying and Multi-Layer Social Graphs", 
Proceedings of the International AAAI Conference on Web and Social Media, 9(1), 377-386, 2021. Disponible en: https://ojs.aaai.org/index.php/ICWSM/article/view/14605. [Darrer accés: 13-abr-2025].

[4] S. Paul, J. Salas i V. Torra, "Edge Local Differential Privacy for Dynamic Graphs", 
In: Modeling Decisions for Artificial Intelligence, M. S. Hossain, A. E. Hassanien y B. Ali, Eds. Cham: Springer Nature Switzerland, 2023. 
Disponible en: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf. [Darrer accés: 13-abr-2025].

[5] B. Ruan, J. Gan, H. Wu, i A. Wirth, "Dynamic Structural Clustering on Graphs", 
arXiv preprint arXiv:2108.11549, 2021. Disponible en: https://arxiv.org/abs/2108.11549. [Darrer accés: 1-mar-2025].

[6] E. Castrillo, E. León, i J. Gómez, "Dynamic Structural Similarity on Graphs", 
arXiv preprint arXiv:1805.01419, 2018. Disponible en: https://arxiv.org/abs/1805.01419. [Darrer accés: 1-mar-2025].

[7] B. Rozemberczki, Awesome Community Detection - Temporal Networks. GitHub. 
Disponible en: https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/temporal.md. [Darrer accés: 26-feb-2025]. 

[8] P. Sarkar, D. Chakrabarti i M. Jordan, "Nonparametric Link Prediction in Dynamic Networks", arXiv preprint arXiv:1206.6394, 2012.
Disponible en: https://arxiv.org/pdf/1206.6394. [Darrer accés: 2-mar-2025].

[9] X. Li, N. Du, H. Li, K. Li, J. Gao i A. Zhang, "Deep Learning Approach to Link Prediction in Dynamic Networks", SIAM, 2014.
Disponible en: https://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.33. [Darrer accés: 2-mar-2025].

[10] J. You, T. Du, J. Leskovec "ROLAND: Graph Learning Framework for Dynamic Graphs", Conference on Knowledge Discovery and Data Mining, 2022. 
Disponible en: https://arxiv.org/pdf/2208.07239 [Darrer accés: 6-mar-2025] 

[11] "Havel–Hakimi algorithm", Wikipedia, l'enciclopèdia lliure. Disponible en: https://en.wikipedia.org/wiki/Havel%E2%80%93Hakimi_algorithm. [Darrer accés: 13-abr-2025].

[12] "Erdős–Gallai theorem", Wikipedia, l'enciclopèdia lliure. Disponible en: https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Gallai_theorem. [Darrer accés: 14-abr-2025].

[13] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0228728 -- 19/04

[14] https://web.eecs.umich.edu/~dkoutra/papers/DeltaCon_KoutraVF_withAppendix.pdf?utm_source=chatgpt.com -- 19/04


> F. Beck, M. Burch, S. Diehl i D. Weiskopf, "The State of the Art in Visualizing Dynamic Graphs", 
> Eurographics Conference on Visualization (EuroVis) - State of The Art Report, 2014. 
> Disponible en: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf. 
> [Darrer accés: 13-gen-2025].

> P. Agarwal, R. Verma, A. Agarwal i T. Chakraborty, "DyPerm: Maximizing Permanence for Dynamic Community Detection", 2018.
> Disponible en: https://arxiv.org/pdf/1802.04593. [Darrer accés: 26-feb-2025].

> S. Boudebza, R. Cazabet, O. Nouali i F. Azouaou, "Detecting Stable Communities in Link Streams at Multiple Temporal Scales", 2019. 
> Disponible en: https://arxiv.org/pdf/1907.10453. [Darrer accés: 26-feb-2025].

> R. Clark, G. Punzo i M. Macdonald, "Network Communities of Dynamical Influence". Sci Rep 9, 17590, 2019. 
> Disponible en: https://www.nature.com/articles/s41598-019-53942-4. [Darrer accés: 26-feb-2025]. 

> G. Rossetti, Awesome Network Analysis. GitHub. 
> Disponible en: https://github.com/GiulioRossetti/awesome-network-analysis. [Darrer accés: 26-feb-2025].

> K. Zhao, C. Guo, Y. Cheng, P. Han, M. Zhang i B. Yang,  "Multiple Time Series Forecasting with
> Dynamic Graph Modeling".  Proceedings of the VLDB Endowment, 17(4), 753-765, 2023. 
> Disponible en: https://vbn.aau.dk/ws/portalfiles/portal/698843020/3636218.3636230.pdf. [Darrer accés: 26-feb-2025].

