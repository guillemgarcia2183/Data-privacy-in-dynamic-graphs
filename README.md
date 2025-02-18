# Privacy and community detection in dynamic graphs
## Anonymization of temporal graphs using privacy methods

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

* Estudiar el cost que té aplicar el mètode d'anonimització, tant en temps, la pèrdua de dades que pot comportar, o bé la seva utilitat. 

* Analitzar en quins casos pot anar millor els algoritmes desenvolupats segons els resultats que ens dona.

* Comparar diferents algoritmes de detecció de comunitats en grafs dinàmics, tant si se li aplica un mètode de protecció o no.

## Objectius secundaris 

* Fer un mètode per predir el graf que conformarà el següent instant de temps (les arestes que es connecten en t+1).

* Intentar visualitzar a mode d'animació com es protegeixen les dades, o bé la comparació de detecció de comunitats durant el temps.

## Referències

SNAP, "CollegeMsg: University of California social network graph", Stanford University. [Online]. Disponible en: https://snap.stanford.edu/data/CollegeMsg.html. [Darrer accés: 11 de gener del 2025].

J. Salas i V. Torra, "A General Algorithm for k-anonymity on Dynamic Databases," Data Privacy Management, Cryptocurrencies and Blockchain Technology, Barcelona, Espanya, 2018. [Online]. Disponible en: https://link.springer.com/chapter/10.1007/978-3-030-00305-0_28. [Darrer accés: 9 de gener del 2025].

S. Paul, J. Salas i V. Torra, "Edge Local Differential Privacy for Dynamic Graphs", In: Modeling Decisions for Artificial Intelligence, M. S. Hossain, A. E. Hassanien y B. Ali, Eds. Cham: Springer Nature Switzerland, 2023. [Online]. Disponible en: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf. [Darrer accés: 2 de febrer del 2025]. 

F. Beck, M. Burch, S. Diehl i D. Weiskopf, "The State of the Art in Visualizing Dynamic Graphs", Eurographics Conference on Visualization (EuroVis) - State of The Art Report, 2014. [Online]. Disponible en: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf. [Darrer accés: 9 de gener del 2025].

J. Salas, V. Torra, "Differentially private graph publishing and randomized
response for collaborative filtering", In: Proceedings of the 17th International
Joint Conference on e-Business and Telecommunications, ICETE 2020 - Volume
2: SECRYPT, Lieusaint, Paris, France, 8–10 July 2020, pp. 415–422. ScitePress
(2020). [Online]. Disponible en: https://www.diva-portal.org/smash/get/diva2:1534357/FULLTEXT01.pdf. [Darrer accés: 6 de febrer del 2025].

https://github.com/benedekrozemberczki/awesome-community-detection **[Referència per dir quin és l'estat de l'art]**

* DYNAMIC COMMUNITY DETECTION FOR EVOLVING NETWORKS: https://arxiv.org/pdf/1905.01498

* Network communities of Dynamical Influence: https://www.nature.com/articles/s41598-019-53942-4

* Temporal methods: https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/temporal.md

* Spectral methods: https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/spectral.md#spectral-methods

* Deep learning methods: https://github.com/benedekrozemberczki/awesome-community-detection/blob/master/chapters/deep_learning.md

Enron -- https://snap.stanford.edu/data/email-Enron.html -- 18/02

reptilia-tortoise-network-lm: Easy (45, 134) -- https://networkrepository.com/reptilia-tortoise-network-lm.php -- 18/02

aves-sparrow-social: Easy (52, 516) -- https://networkrepository.com/aves-sparrow-social.php -- 18/02

insecta-ant-colony5: Medium (152, 194K) -- https://networkrepository.com/insecta-ant-colony5.php -- 18/02

ia-facebook-wall-wosn-dir: Hard (42.4K, 877K) -- https://networkrepository.com/ia-facebook-wall-wosn-dir.php -- 18/02

https://github.com/GiulioRossetti/awesome-network-analysis

https://projects.csail.mit.edu/dnd/DBLP/

https://catalog.caida.org/dataset/as_rank