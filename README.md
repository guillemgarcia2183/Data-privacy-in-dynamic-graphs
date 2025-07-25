# Privacy in dynamic graphs

## Abstract
>Data privacy is becoming increasingly relevant across all fields. In particular, it is a
>growing concern in networks that generate large volumes of data and constant interactions, such
>as transaction or communication networks. These networks are often modeled as graphs, which
>can evolve over time with changes in nodes and relationships. This temporal aspect introduces
>new privacy challenges and makes it harder to protect sensitive information. For this reason, two
>data protection methods are presented, based on k-anonymity and Differential privacy. To evaluate
>the implemented algorithms, similarity, utility, and structural metrics are used, specially adapted to
>compare temporal graphs. Specifically, one of the goals is to analyze whether the protected graphs
>preserve their features when community detection algorithms are applied, using TSCAN for this task.

## Installation and how to run  
Follow the steps below to clone the repository, install the dependencies, and run the code:

```bash
git clone https://github.com/guillemgarcia2183/TFG-Dynamic-Graphs.git
pip install -r requirements.txt
python3 code/main.py
```

## Datasets
Here you can find the datasets used to apply the privacy methods and evaluate their metrics.

<div align="center">
  
| Name | # Vertices | # Edges | Is directed | Has weight | # Snapshots |
|-----------|-----------|-----------|-----------|-----------|-----------|
| Aves-sparrow | 52 | 516 | False | True | 2 | 
| Mammalia-voles | 1480 | 4569 | False | True | 61 | 
| Insecta-ant | 152 | 194K | False | True | 41 |
| Enron-Employees | 151 | 50.5K | True | True | 16067 | 
| CollegeMsg | 1899 | 59.8K | True | False | 58911 | 

</div>

> [!NOTE]
> You can add your own dataset, but it must be in `.txt` or `.json` format and include the following columns. The `weight` column is optional and only used for graphs that include edge weights.

<div align="center">
  
| From_node | To_node | Weight (optional) | Timestamp |
|-----------|---------|-----------|-------------------|

</div>










