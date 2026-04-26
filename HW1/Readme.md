# AMI Knowledge Graph Project

This project builds a knowledge graph for the Faculty of Applied Mathematics and Informatics (AMI), Ivan Franko National University of Lviv, from public AMI web materials.

## Graph schema

### Nodes

- `Faculty`
- `OrgUnit`
- `Department`
- `Laboratory`
- `AdministrativeUnit`
- `Person`
- `Course`

### Relationships

- `(:Faculty)-[:HAS_UNIT]->(:OrgUnit)`
- `(:Person)-[:AFFILIATED_WITH]->(:OrgUnit)`
- `(:Person)-[:TEACHES]->(:Course)`

### Why this schema

AMI website sections are not all departments. Some sections are administrative units and some are laboratories. The loader classifies sections before writing them to Neo4j.

## Recommended notebook workflow

### Notebook 1

Use 01_kg_construction.ipynb for:
- GraphAcademy summary
- source explanation
- schema explanation
- scraping and preprocessing
- Neo4j loading
- validation

### Notebook 2

Use 02_retrieval_demo.ipynb for:
- LangChain demo
- GraphRAG demo
- example QA scenarios
- limitations and improvements
