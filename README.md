# GMDA-TDAForFinancial

## Project
The goal of this project was to analyze the evolutionof daily returns of four 
major US stock markets indices(DowJones, Nasdaq, Russell2000, SP500) over the 
per-iod 1987 2016 using persistent homology following theapproach proposed by 
M. Gidea and Y. Katz in [1].


The idea was also to develop a Command line 
interface under Python allowing to reproduce the results of our approach.

## Installation
The whole module is based on the Gudhi library [2]. 
The graphics are displayed using matplotlib [3].
Since Gudhi is not located in the pip, 
we advise you to use Anaconda.

To install the requirements, you can execute the following command:
```bash
conda install --file='requirements.txt' -y
```

We also advise you to create a new environment:
```bash
conda create --name='GMDA-env' -y
conda activate GMDA-env
```
You can locate the environment using the command
```bash
conda env list 
```

## Command Line Interface

### Demo
You can run the demo script to get a 
preview of the contents of the package.
```bash
chmod +x ./demo.sh
./demo.sh
```

You can locate the environment using the command
```bash
conda env list 
```

### Documentation
`Note your python interpreter linked to anaconda "pyconda".
You can find it by locating your environment. 
The interpreter is usually found in the bin/python folder`

The general idea of the command is to use the "manage.py" file.
```bash
pyconda manage.py <\command>
```
At the first launch, the program will download the datasets from the internet. 

#### dataset
```bash
pyconda manage.py dataset <\subcommand>
```
| subcommand     | explaination            | options   |
| :----------:   | :----------:             | :----------: |
|  visualise     | visualise the dataset   | --log <br/> --save    |

#### landscape

#### norm

#### bottleneck



## Reference 
[1]: Marian Gidea and Yuri Katz. Topological data analysisof 
financial time series : Landscapes of crashes.PhysicaA 
: Statistical Mechanics and its Applications, 491 :820– 834, 2018 \
[2]: https://gudhi.inria.fr \
[3]: https://matplotlib.org


TODO ajouter Gudhi
