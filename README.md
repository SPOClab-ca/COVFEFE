# COVFEFE
COre Variable Feature Extraction Feature Extractor

A fast, multi-threaded tool for running various feature extraction pipelines. A pipeline is a directed acyclic graph 
where each node is a processing task that sends it's output to the next node in the graph.

Nodes are defined in ```nodes/``` and pipelines in ```pipelines/```.

An example pipeline is ```opensmile_is10_lld``` defined in ```pipelines/pipelines.py```. The function that creates the pipeline 
is decorated using ```@pipeline_registry``` which adds it to registry containing all pipelines.

To execute a pipeline, simply run

```bash
python covfefe.py -i path/to/in/folder -o /path/to/put/out/files -p pipeline_name
```

where pipeline_name is the name of function that's been added to the registry (for example, ```opensmile_is10_lld```)

to run with multiple threads (ex. 8 threads):
```bash
python covfefe.py -i path/to/in/folder -o /path/to/put/out/files -p pipeline_name -n 8
```

Running ```python covfefe.py --help``` will print help options and give a list of available pipelines to run.

## Install
First download covfefe and setup a virtual environment:
```bash
git clone https://github.com/SPOClab-ca/COVFEFE.git
cd COVFEFE
virtualenv -p python3 .venv
```

Activate the virtual environment
```bash
source .venv/bin/activate
```

Install python libraries:
```bash
pip install -r requirements.txt
```

Install nltk packages, if not already installed
```bash
python -c "import nltk; deps=['cmudict', 'wordnet_id', 'punkt', 'wordnet']; [nltk.download(d) for d in deps];"
```


A script is provided that will download and setup dependencies. Before running this script, you should have openSMILE
installed. You can find instructions on the [openSMILE website](https://audeering.com/technology/opensmile/)
```bash
./setup.sh /path/to/put/downloaded/files
```
This script will ask you to enter the path to the openSMILE source. This is the path to the extracted zip or tar file, 
not the smilExtract binary. 

Next the setup script downloads various dependencies (requires 1.6 GB of disk space) and creates a file called 
`config.ini` which stores paths to the dependencies. When covfefe is run, it will try to find it's dependencies 
from environment variables first, then this config file.  

## Default Pipelines
<table>
    <tr>
        <th>Pipeline</th>
        <th>Input data</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>split_speech_eaf</td>
        <td>.wav, .eaf</td>
        <td>Reads in wav files and eaf files (with same name except for extension) from input folder and 
        splits each annotated segment into its own wav file</td>
    </tr>
    <tr>
        <td>split_speech_txt</td>
        <td>.wav, .txt</td>
        <td>Same as split_speech_eaf except uses tab separated .txt files as annotations (start \t end \t annotation)</td>
    </tr>
    <tr>
        <td>opensmile_is10_lld</td>
        <td>.wav</td>
        <td>Computes openSMILE IS10 low level descriptors for each wav file in the input folder</td>
    </tr>
    <tr>
        <td>matlab</td>
        <td>.wav</td>
        <td>Computes matlab acoustic features for each wav file in the input folder (very slow)</td>
    </tr>
    <tr>
        <td>lex</td>
        <td>.txt</td>
        <td>Computes lexicosyntactic features for each txt file in the input</td>
    </tr>
    <tr>
        <td>main</td>
        <td>.wav, .eaf</td>
        <td>Computers IS10 (both low level descripors and full file summaries) for each wav file in the input. If the 
        file has an associated .eaf file, it will split all annotations into individual files and compute IS10 feautures 
        on the isolated .wav files</td>
    </tr>
</table>
 
## Custom pipelines
You can create your own custom nodes and pipelines. For example, if you wanted to create a pipeline that computed one
feature vector for each wav file, you could copy the the opensmile_is10_lld pipeline and change the output flag to 
'-csvoutput'. 

Any pipelines added to the `pipelines/` folder and decorated with `@pipeline_registry` will be automatically discovered
and available through the CLI.

If you would like to make your custom pipelines and nodes available for others to use, please feel free to make a pull request.

## Optional dependencies

##### Receptiviti LIWC
LIWC2015 features from [receptiviti](https://www.receptiviti.ai/liwc-api-get-started) can also be added to the output. Simply copy `secrets.py.example`, rename it to `secrets.py` and fill in your api key. 

##### Matlab
If you have matlab installed on your system, you can install the MATLAB Engine API. As long as your matlab script 
takes as input a path the the input file and a path specifying where to save the output, you should be able to create
a pipeline that uses the `nodes.matlab.MatlabRunner` to call your matlab script.

##### ANEW
If you have access to the ANEW2010 dictionary, you can put the ANEW2010All.txt file in path you gave to the setup script
to add additional features to lexicosyntactic output. You will also need to add
`export path_to_anew=/path/to/dependencies/ANEW2010All.txt` to `env.sh`.

##### RST Discourse Treebank
Similar to ANEW, you can put the RST treebank data in the dependency folder and the following lines to 'env.'
```bash
export path_to_rst_python=path/to/deps/RST/rstenv/bin/python
export path_to_rst=path/to/deps/RST/src/
```





