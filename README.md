# TFG: Privacy in dynamic graphs
## Anonymization of temporal graphs using privacy methods

> For this project, we consider dynamic graphs. That 
> is, graphs which have a temporal component. In such graphs we might find
> vertices or even nodes which appear or disappear in a temporal basis.
> Anonymizing these graphs introduces important challenges not found in
> more classical static graphs. In this project, we aim to explore
> possible solutions for the anonymization of temporal graphs by relaying
> in well known privacy models such as k-anonymity and differential privacy.

## Objectius (no definitius*)

* Estudiar de manera teòrica què és un graf dinàmic, i quina aplicació té en l'actualitat.

* Quins són els mètodes que es poden fer servir per aplicar privacitat en un graf dinàmic (estat de l'art), i quins són els que s'estudiaràn. En principi, es té com a objectiu utilitzar els conceptes de K-Anonimity i Differential Privacy. 

* Pels mètodes que s'utilitzen, explicar tot el seu funcionament i què aporta en termes de privacitat.

* Implementar el mètode d'anonimització en diferents datasets de dades (orientat en grafs), des de casos senzills fins a problemes més complexos.

* Estudiar el cost que té aplicar el mètode d'anonimització, tant en temps, com també la pèrdua de dades que pot comportar. Analitzar en quins casos pot anar millor els algoritmes desenvolupats.

* Intentar visualitzar a mode d'animació com es va aplicant el mètode (?)

## Requeriments 

## Example of usage

## Referències

SNAP, "CollegeMsg: University of California social network graph", Stanford University. [Online]. Disponible en: https://snap.stanford.edu/data/CollegeMsg.html. [Darrer accés: 11 de gener del 2025].

SNAP, "Wiki-talk-temporal: Wiki-talk social network graph", Stanford University. [Online]. Disponible en: https://snap.stanford.edu/data/wiki-talk-temporal.html. [Darrer accés: 11 de gener del 2025].

J. Salas i V. Torra, "A General Algorithm for k-anonymity on Dynamic Databases," Data Privacy Management, Cryptocurrencies and Blockchain Technology, Barcelona, Espanya, 2018. [Online]. Disponible en: https://link.springer.com/chapter/10.1007/978-3-030-00305-0_28. [Darrer accés: 9 de gener del 2025].

S. Paul, J. Salas i V. Torra, "Edge Local Differential Privacy for Dynamic Graphs", In: Modeling Decisions for Artificial Intelligence, M. S. Hossain, A. E. Hassanien y B. Ali, Eds. Cham: Springer Nature Switzerland, 2023. [Online]. Disponible en: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf. [Darrer accés: 2 de febrer del 2025]. 

F. Beck, M. Burch, S. Diehl i D. Weiskopf, "The State of the Art in Visualizing Dynamic Graphs", Eurographics Conference on Visualization (EuroVis) - State of The Art Report, 2014. [Online]. Disponible en: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf. [Darrer accés: 9 de gener del 2025].

J. Salas, V. Torra, "Differentially private graph publishing and randomized
response for collaborative filtering", In: Proceedings of the 17th International
Joint Conference on e-Business and Telecommunications, ICETE 2020 - Volume
2: SECRYPT, Lieusaint, Paris, France, 8–10 July 2020, pp. 415–422. ScitePress
(2020). [Online]. Disponible en: https://www.diva-portal.org/smash/get/diva2:1534357/FULLTEXT01.pdf. [Darrer accés: 6 de febrer del 2025].