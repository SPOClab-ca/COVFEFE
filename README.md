# COVFEFE
COre Variable Feature Extraction Feature Extractor

## Table of Contents
* [Simple description](#simple-description)
* [Detailed description](#detailed-description)
* [Installation instructions](#install)
* [Available Pipelines](#available-pipelines)
* [Custom Pipelines](#custom-pipelines)
* [Optional dependencies](#optional-dependencies)

## Simple Description
COVFEFE is a tool for feature extraction.  
Given a folder containing your data, it will compute features for each 
file in the input folder. It currently supports acoustic features on audio data and lexical, syntactic and pragmatic features on text data (english and chinese), but can be extended to other features and data types (feel free to
make a pull request if you would like to add more). 

As an example, given an input folder with two audio files and two text files.
```
input_data
├── file1.txt
├── file1.wav
├── file2.txt
└── file2.wav
```
To extract acoustic features for all the wav files:
```bash
python covfefe.py -i input_data -o output_folder -p opensmile_is10
```
To extract lexicosyntactic features on all the txt files:
```bash
python covfefe.py -i input_data -o output_folder -p lex
```
This will create an output folder with all the features:

```
output_folder
├── is10
│   ├── file1.csv
│   └── file2.csv
└── lexicosyntactic
    ├── file1.csv
    └── file2.csv
```


## Detailed Description
COVFEFE is a a fast, multi-threaded tool for running various feature extraction pipelines. A pipeline is a directed acyclic graph 
where each node is a processing task that sends it's output to the next node in the graph.

Nodes are defined in ```nodes/``` and pipelines in ```pipelines/```.

An example pipeline is ```opensmile_is10_lld``` defined in ```pipelines/pipelines.py```. 

```python
@pipeline_registry
def opensmile_is10_lld(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = audio.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")


    p = ProgressPipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p
```

The function is decorated using ```@pipeline_registry``` which adds it to registry containing all pipelines. When called, a 
pipeline function will be provided an input folder, output folder and number of threads as parameters. These parameters are 
used to configure the pipeline. The `opensmile_is10_lld` function shown above first creates a node to find all the files in 
the input folder that have a `.wav` extension. The second node it creates is an `OpenSmileRunner`, which is defined in the 
`nodes.audio` package. This node passes its input to openSMILE (https://audeering.com/technology/opensmile/), a feature 
extraction tool. Some common nodes (such as converting `wav` to `mp3`, resampling audio, calling matlab functions or shell scripts)
are provided and users can define their own nodes.

After defining the nodes, the `opensmile_is10_lld` function creates a pipeline using the `|` operator. This is inspired by the 
unix pipe, and simply means that the output of the left node is passed to the right node. The right hand side of the operator 
can be a list of nodes, in which case the input from the left side is passed to all nodes in the list. 

The way covfefe is set up, each node accepts as input a file path and outputs a file path. Standardizing this makes it easier
to create new nodes and pipelines that are interoperable.

After creating the pipeline, `p`, the pipeline function returns it. The pipeline so far has only been defined and not executed.
It will be executed by the main function in `covfefe.py`, which is the script you can use to call different pipelines. 
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

## Available Pipelines
<table>
    <tr>
        <th>Pipeline</th>
        <th>Input data</th>
        <th>Description</th>
        <th>Requirements</th>
    </tr>
    <tr>
        <td>split_speech_eaf</td>
        <td>.wav, .eaf</td>
        <td>Reads in wav files and eaf files (with same name except for extension) from input folder and 
        splits each annotated segment into its own wav file</td>
        <td></td>
    </tr>
    <tr>
        <td>split_speech_txt</td>
        <td>.wav, .txt</td>
        <td>Same as split_speech_eaf except uses tab separated .txt files as annotations (start \t end \t annotation)</td>
        <td></td>
    </tr>
    <tr>
        <td>opensmile_is10</td>
        <td>.wav</td>
        <td>Computes openSMILE IS10 features describing an entire wav file for each wav file in the input folder</td>
        <td>openSMILE is installed and OPENSMILE_HOME is set in `config.ini`</td>
    </tr>
    <tr>
        <td>opensmile_is10_lld</td>
        <td>.wav</td>
        <td>Computes openSMILE IS10 low level descriptors for each wav file in the input folder</td>
        <td>openSMILE is installed and OPENSMILE_HOME is set in `config.ini`</td>
    </tr>
    <tr>
        <td>praat_syllable_nuclei</td>
        <td>.wav</td>
        <td>Runs Praat script that computes syllable nuclei features</td>
        <td>praat is installed</td>
    </tr>
    <tr>
        <td>matlab</td>
        <td>.wav</td>
        <td>Computes matlab acoustic features for each wav file in the input folder (very slow)</td>
        <td>Matlab engine for python is installed</td>
    </tr>
    <tr>
        <td>lex</td>
        <td>.txt</td>
        <td>Computes lexicosyntactic features for each txt file in the input</td>
        <td>All dependencies were downloaded using the `setup.sh` script `config.ini` was correctly generated</td>
    </tr>
    <tr>
        <td>lex_chinese</td>
        <td>.txt</td>
        <td>Computes lexicosyntactic features for Chinese text files</td>
        <td>Same as the lex pipeline</td>
    </tr>
    <tr>
        <td>kaldi_asr</td>
        <td>.wav</td>
        <td>Runs automatic speech recognition on all wav files using kaldi. Wav files will be reasmpled to 8KHz.</td>
        <td>Kaldi is installed and compiled and the `aspire` example is setup.</td>
    </tr>
    <tr>
        <td>main</td>
        <td>.wav, .eaf</td>
        <td>Computers IS10 (both low level descripors and full file summaries) for each wav file in the input. If the 
        file has an associated .eaf file, it will split all annotations into individual files and compute IS10 feautures 
        on the isolated .wav files</td>
        <td>opensmile + lex + kaldi</td>
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
`path_to_anew=/path/to/dependencies/ANEW2010All.txt` to `config.ini`.

##### RST Discourse Treebank
Similar to ANEW, you can put the RST treebank data in the dependency folder and the following lines to 'config.ini'
```bash
path_to_rst_python=path/to/deps/RST/rstenv/bin/python
path_to_rst=path/to/deps/RST/src/
```




