# COVFEFE
COre Variable Feature Extraction Feature Extraction

A tool for running various feature extraction pipelines. A pipeline is a directed acyclic graph where each node is a 
processing task that sends it's ouptut to the next node in the graph.

Nodes are defined in ```nodes/``` and pipelines in ```pipelines/```.

An example pipeline is ```opensmile_is10_lld``` defined in ```opensmile_is10```. The function that creates the pipeline 
is decorated using ```@pipeline_registry``` which adds it to registry containing all pipelines.

To execute a pipeline, simply run

```bash
python covfefe.py -i path/to/in/folder -o /path/to/put/out/files -p pipeline_name
```

where pipeline_name is the name of function that's been added to the registry (for example, opensmile_is10_lld)
