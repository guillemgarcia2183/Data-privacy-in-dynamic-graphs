# TFG: Privacy in dynamic graphs

## Proposta 
For this project, we consider dynamic graphs. That
is, graphs which have a temporal component. In such graphs we might find
vertices or even nodes which appear or disappear in a temporal basis.
Anonymizing these graphs introduces important challenges not found in
more classical static graphs. In this project, we aim to explore
possible solutions for the anonymization of temporal graphs by relaying
in well known privacy models such as k-anonymity and differential privacy.

La idea d'aquest projecte seria mirar d'aplicar conceptes coneguts de
privacitat a graphs estàtics en aquest tipus de graphs temporals. Un
primer pas seria veure com fer que el graph temporal complís la
propietat de k-anonymitat pel grau dels nodes. Això, en considerar
l'aspecte temporal passa a ser una mica complex, ja que cal tenir en
compte tota l'evolució temporal del graf. Al projecte nosaltres us
donaríem una primera proposta de com fer-ho, però òbviament estaria
obert a les vostres propostes o solucions.

Es tractaria d'implementar aquest mètode d'anonimització i aplicar-ho a
algun dataset concret, com per exemple el dels mails de enron (abans
primer es podrien fer proves amb altres més petits clar).

## Metodologia

Per una banda, s'intentarà resoldre de manera teòrica el dilema que es té en aplicar privacitat dintre d'un graf dinàmic. Seguidament,
s'implementarà el mètode d'anonimització en diferents conjunts de dades per posar en pràctica el problema. Això es farà a partir de programació Python amb .... llibreries de suport. 

## Objectius

* Estudiar de manera teòrica què és un graf dinàmic, i l'importància que té en el nostre dia a dia.

* Quins són els mètodes que es poden fer servir per aplicar privacitat en un graf dinàmic (estat de l'art), i quins són els que s'utilitzaràn. En principi, es té com a objectiu utilitzar K-Anonimity i Differential Privacy a nivell pràctic. 

* Pels mètodes que s'utilitzen, explicar tot el seu funcionament i què aporta en termes de privacitat.

* Implementar el mètode d'anonimització en diferents datasets, des de casos senzills fins a problemes complexos.

* Estudiar el cost que té aplicar el mètode d'anonimització, tant en temps d'execució, com també la pèrdua de dades que pot comportar.

* Intentar visualitzar a mode d'animació com es va aplicant el mètode (?)

## Referències
Visualització de grafs dinàmics: https://www.visus.uni-stuttgart.de/documentcenter/forschung/visualisierung_und_visual_analytics/eurovis14-star.pdf -- 09/01/2025

K-anonimity en dades dinàmiques: https://link.springer.com/chapter/10.1007/978-3-030-00305-0_28 -- 11/01/2025

University of California social network graph: https://snap.stanford.edu/data/CollegeMsg.html -- 11/01/2025

Wiki-talk social network graph: https://snap.stanford.edu/data/wiki-talk-temporal.html -- 11/01/2025

Edge Local Differential Privacy for Dynamic Graphs: https://link.springer.com/content/pdf/10.1007/978-981-99-5177-2_13.pdf -- 25/01/2025

