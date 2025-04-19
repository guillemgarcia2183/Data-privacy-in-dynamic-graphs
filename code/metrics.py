
class Metrics:
    __slots__ = ('originalGraphs', 'protectedGraphs')
    """Classe que computa i visualitza les mètriques entre dos grafs.
    """
    def __init__(self, originalGraphs, protectedGraphs):
        """Inicialitza la classe Metrics amb els grafs originals i protegits.

        Args:
            originalGraphs (List): Grafs originals 
            protectedGraphs (List): Grafs protegits per un algorisme de protecció
        """
        self.originalGraphs = originalGraphs
        self.protectedGraphs = protectedGraphs
    
    def edgeIntersection(self, e1, e2):
        """Calcular la intersecció d'arestes de dos grafs

        Args:
            e1, e2 (set): Conjunt d'arestes d'un graf

        Returns:
            int: Nombre d'elements en la intersecció
        """
        intersection = e1 & e2
        # print(f"Intersection: {intersection}")
        return len(intersection)
    
    def edgeUnion(self, e1, e2):
        """Calcular l'unió d'arestes de dos grafs

        Args:
            e1, e2 (set): Conjunt d'arestes d'un graf

        Returns:
            int: Nomnre d'elements en l'unió
        """
        union = e1 | e2
        # print(f"Union: {union}")
        return len(union)

    def jaccardCoefficient(self, g1, g2):
        # En cas de no ser dirigit els grafs, s'ha de descartar les arestes duplicades
        if g1.is_directed():
            edgesG1 = set(g1.edges())
            edgesG2 = set(g2.edges())
        else:
            edgesG1 = set(tuple(sorted(edge)) for edge in g1.edges())
            edgesG2 = set(tuple(sorted(edge)) for edge in g2.edges())
        
        # print(f"Edges G1: {edgesG1}, Edges G2: {edgesG2}")

        intersection = self.edgeIntersection(edgesG1, edgesG2)
        union = self.edgeUnion(edgesG1, edgesG2)
 
        print()

        # Quan un dels dos grafs són buits o els dos són buits
        if union == 0:
            return 1.0 if intersection == 0 else 0.0  

        # Calculem el coeficient de Jaccard i el retornem en forma de percentatge
        return (intersection / union)*100

    def deltaConnectivity(self, g1, g2):
        pass
    
    def spectralSimilarity(self, g1, g2):
        pass

    def centralityCorrelation(self, g1, g2, function):
        pass
    
    def computeMetrics(self):
        pass

    def visualizeMetrics(self):
        pass