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

Com hem esmentat en l'apartat 1, es volen aplicar diferents mètodes de privacitat. 
Els casos d'estudi son diversos tipus de datasets trobats en [1,2]. 
La llista de datasets que es conté per defecte és la de a continuació:

<div align="center">
  
| Name | # Vertices | # Edges | Is directed | Has weight 
|-----------|-----------|-----------|-----------|-----------|
| Aves-sparrow | 52 | 516 | False | True |
| Reptilia-tortoise | 45 | 134 | False | False |
| Insecta-ant | 152 | 194K | False | True |
| CollegeMsg | 1899 | 59.8K | True | False |
| IA-Facebook | 42.4K | 877K | True | True |

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

Les maneres de protegir grafs són diverses, com per exemple fer que els nodes siguin indistinguibles entre ells a partir dels seus atributs, afegir soroll, 
amb estratègies de xifratge, etc. Dintre dels grafs dinàmics ve a ser el mateix, però, s'ha de mantenir protegit tant tot el conjunt sencer com de forma individual. 
Les estratègies que s'utilitzarà serà principalment el K-Anonymity [3] i el Edge-Local Differential Privacy [4] en grafs dinàmics, [Estat de l'art d'atacs(?)]. 
En [5] es pot observar diferents mètodes per a la detecció de comunitats dintre de grafs que canvien durant el temps. L'itenció és utilitzar-ne gran part d'aquests per fer comparativa respecte els grafs originals i protegits.

## 4. Referències

[1] J. Leskovec, Stanford Network Analysis Project (SNAP). Disponible en: https://snap.stanford.edu/index.html. [Darrer accés: 26-feb-2025].

[2] Ryan A. Rossi i Nesreen K. Ahmed, The Network Data Repository with Interactive Graph Analytics and Visualization, 2015. 
Disponible en: https://networkrepository.com/dynamic.php. [Darrer accés: 26-feb-2025].

[3] L. Rossi, M. Musolesi i A. Torsello, "On the k-Anonymization of Time-Varying and Multi-Layer Social Graphs", Proceedings of the International AAAI Conference on Web and Social Media, 9(1), 377-386, 2021. Disponible en: https://ojs.aaai.org/index.php/ICWSM/article/view/14605. [Darrer accés: 26-feb-2025].

[4] S. Paul, J. Salas i V. Torra, "Edge Local Differential Privacy for Dynamic Graphs", 
In: Modeling Decisions for Artificial Intelligence, M. S. Hossain, A. E. Hassanien y B. Ali, Eds. Cham: Springer Nature Switzerland, 2023. 
Disponible en: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf. [Darrer accés: 2-feb-2025].

[5] B. Rozemberczki, Awesome Community Detection - Temporal Networks. GitHub. 
Disponible en: https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/temporal.md. [Darrer accés: 26-feb-2025]. 

> F. Beck, M. Burch, S. Diehl i D. Weiskopf, "The State of the Art in Visualizing Dynamic Graphs", 
> Eurographics Conference on Visualization (EuroVis) - State of The Art Report, 2014. 
> Disponible en: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf. 
> [Darrer accés: 13-gen-2025].

> J. Salas, V. Torra, "Differentially private graph publishing and randomized
> response for collaborative filtering", In: Proceedings of the 17th International
> Joint Conference on e-Business and Telecommunications, ICETE 2020 - Volume
> 2: SECRYPT, Lieusaint, Paris, France, 8–10 July 2020, pp. 415–422. ScitePress
> (2020). Disponible en: https://www.diva-portal.org/smash/get/diva2:1534357/FULLTEXT01.pdf. [Darrer accés: 6-feb-2025].

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

> S. Raskhodnikova i T. Steiner, "Fully Dynamic Graph Algorithms with Edge Differential Privacy", 2024. 
> Disponible en: https://arxiv.org/pdf/2409.17623. [Darrer accés: 26-feb-2025]. 




