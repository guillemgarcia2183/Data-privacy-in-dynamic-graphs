# Privacy in dynamic graphs
## Anonymization and community detection
> For this project, we consider dynamic graphs. That 
> is, graphs which have a temporal component. In such graphs we might find
> vertices or even nodes which appear or disappear in a temporal basis.
> Anonymizing these graphs introduces important challenges not found in
> more classical static graphs. In this project, we aim to explore
> possible solutions for the anonymization of temporal graphs by relaying
> in well known privacy models such as k-anonymity and differential privacy.

## Objectius principals

* Estudiar de manera teòrica què és un graf dinàmic, i quina aplicació té en l'actualitat.

* Quins són els mètodes que es poden fer servir per aplicar privacitat en un graf dinàmic (estat de l'art), i quins són els que s'estudiaràn. En principi, es té com a objectiu utilitzar els conceptes de K-Anonimity i Differential Privacy. 

* Pels mètodes que s'utilitzen, explicar tot el seu funcionament i què aporta en termes de privacitat.

* Implementar el mètode d'anonimització en diferents datasets de dades, des de casos senzills fins a problemes més complexos.

* Investigar sobre la detecció de comunitats dintre de grafs dinàmics. 

* Estudiar el cost que té aplicar el mètode d'anonimització, tant en temps, la pèrdua de dades que pot comportar, o bé la seva utilitat. 

* Analitzar en quins casos pot anar millor els algoritmes desenvolupats segons els resultats que ens dona (comparar algoritmes, els propis paràmetres (k), etc.).

* Comparar diferents algoritmes de detecció de comunitats en grafs dinàmics, tant si se li aplica un mètode de protecció o no.

## Objectius secundaris 

* Fer un mètode per predir el graf que conformarà el següent instant de temps. En intentar fer aquesta predicció, l'objectiu principal és que aquest graf generat també compleixi les mateixes propietats.

* Realitzar una simulació d'un atac a la xarxa, i comparar els algoritmes protegits amb els grafs originals.

* Intentar visualitzar a mode d'animació com es protegeixen les dades, o bé la comparació de detecció de comunitats durant el temps.

## Metodologia

Els casos d'estudi son diversos tipus de datasets trobats en [x,y]. Hi ha varietat de datasets, per tal d'observar les solucions amb diferents volumns de dades i distintes característiques. La llista de datasets que es conté per defecte és la de a continuació:
    * Aves-sparrow dataset (|V| = 52, |E| = 516, weighted, undirected) [x]
    * Reptilia-tortoise dataset (|V| = 45, |E| = 134, unweighted, undirected) [x]
    * Insecta-ant dataset (|V| = 152, |E| = 194K, weighted, undirected) [x]
    * CollegeMsg dataset (|V| = 1899, |E| = 59.8K, unweighted, directed) [x]
    * IA-Facebook dataset (|V| = 42.4K, |E| = 877K, weighted, directed) [x]

El llenguatge de programació que s'ha fet servir principalment ha estat Python v3.13.0, utilitzant les llibreries:
    * NetworkX (v3.2.1): Per a la creació dels grafs a partir de les dades dels datasets. 
    * Pandas (v2.1.4): Per fer conversió de dades en DataFrames. Ús principalment per fer agregacions de dades.
    * Matplotlib (v3.8.2): Utilitzat per visualització de gràfics. Eficient per observar mètriques.
    * Tensorflow (v2.17.0): Per a crear una xarxa neural, en cas de voler fer un forecasting de grafs. 

Com a planificació personal, s'ha realitzat un diagrama de Gantt, estimant en quin moment s'ha de tenir cada possible tasca del treball finalitzada [Annexe]. Les tasques estan dividides segons si forma part a la part conceptual del projecte, o bé si és a nivell pràctic. La part teòrica del treball engloba a la recerca d'informació i la definició dels mètodes. En canvi, l'altre part s'enfoca en la recerca de dades, aplicar els mètodes de protecció i detecció de comunitats estudiats, i finalment fer una anàlisi de mètriques. 

## Estat de l'art

Com a un començament, s'ha fet una recerca de quines són les tècniques que s'utilitzen per protegir grafs dinàmics i també detectar i mantenir l'estructura de les comunitats durant el temps. Per solucionar el problema de privacitat en un graf dinàmic es poden utilitzar (....)

Per la detecció de comunitats, hi ha una llista de mètodes en [x], que s'ha tingut interès principalment en el treball de Network Communities of Dynamical Influence [x], Detecting Stable Communities in Link Streams at Multiple Temporal Scales [x] i DyPerm: Maximizing Permanence for Dynamic Community Detection [x].

## Referències

SNAP, "CollegeMsg: University of California social network graph", Stanford University. [Online]. Disponible en: https://snap.stanford.edu/data/CollegeMsg.html. [Darrer accés: 11 de gener del 2025].

S. Paul, J. Salas i V. Torra, "Edge Local Differential Privacy for Dynamic Graphs", In: Modeling Decisions for Artificial Intelligence, M. S. Hossain, A. E. Hassanien y B. Ali, Eds. Cham: Springer Nature Switzerland, 2023. [Online]. Disponible en: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf. [Darrer accés: 2 de febrer del 2025]. 

F. Beck, M. Burch, S. Diehl i D. Weiskopf, "The State of the Art in Visualizing Dynamic Graphs", Eurographics Conference on Visualization (EuroVis) - State of The Art Report, 2014. [Online]. Disponible en: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf. [Darrer accés: 9 de gener del 2025].

J. Salas, V. Torra, "Differentially private graph publishing and randomized
response for collaborative filtering", In: Proceedings of the 17th International
Joint Conference on e-Business and Telecommunications, ICETE 2020 - Volume
2: SECRYPT, Lieusaint, Paris, France, 8–10 July 2020, pp. 415–422. ScitePress
(2020). [Online]. Disponible en: https://www.diva-portal.org/smash/get/diva2:1534357/FULLTEXT01.pdf. [Darrer accés: 6 de febrer del 2025].

https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/temporal.md **[Referència per dir quin és l'estat de l'art]**

* DyPerm: Maximizing Permanence for Dynamic Community Detection: https://arxiv.org/pdf/1802.04593
* Detecting Stable Communities in Link Streams at Multiple Temporal Scales: https://arxiv.org/pdf/1907.10453
* Network communities of Dynamical Influence: https://www.nature.com/articles/s41598-019-53942-4

Enron -- https://snap.stanford.edu/data/email-Enron.html -- 18/02

reptilia-tortoise-network-lm: -- https://networkrepository.com/reptilia-tortoise-network-lm.php -- 18/02

aves-sparrow-social: -- https://networkrepository.com/aves-sparrow-social.php -- 18/02

insecta-ant-colony5: -- https://networkrepository.com/insecta-ant-colony5.php -- 18/02

ia-facebook-wall-wosn-dir: -- https://networkrepository.com/ia-facebook-wall-wosn-dir.php -- 18/02

https://github.com/GiulioRossetti/awesome-network-analysis

https://projects.csail.mit.edu/dnd/DBLP/

https://vbn.aau.dk/ws/portalfiles/portal/698843020/3636218.3636230.pdf

https://catalog.caida.org/dataset/as_rank

https://arxiv.org/pdf/2409.17623

https://www.researchgate.net/profile/Yitao-Duan-2/publication/220556834_Privacy_Preserving_Link_Analysis_on_Dynamic_Weighted_Graph/links/00b4952776889addee000000/Privacy-Preserving-Link-Analysis-on-Dynamic-Weighted-Graph.pdf

Diagrama de Gantt: https://garciaguillemdausas-team-company.monday.com/boards/1836413048/


