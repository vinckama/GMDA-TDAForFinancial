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

`This documentation is not meant to be exhaustive, however, 
it gives a good idea of the contents of the package. You 
find details on the command arguments using the '-h' argument that you can associate with any
 what an command.`
#### Dataset
Access to the dataset
```bash
pyconda manage.py dataset <\subcommand>
```
| subcommand     | explaination            | arguments   |
| :----------:   | :----------:             | :----------: |
|  visualise     | visualise the dataset   | --log <br/> --save    |

#### Landscape
Access to the landscape persistence
```bash
pyconda manage.py landscape <\subcommand>
```
| subcommand     | explaination            | arguments   |
| :----------:   | :----------:             | :----------: |
|  visualise     | plot the landscape graphs  | -w_size <br/> --end_date <br/> --save    |
|  get     | get the persistence tree and the landscape   | -w_size <br/> --end_date    |
|  clean     | clean the hidden working database   |  |

#### Norm
Access to the norm L1 & L2 of the persistence
```bash
pyconda manage.py norm <\subcommand>
```
| subcommand     | explaination            | arguments   |
| :----------:   | :----------:             | :----------: |
|  visualise     | plot the norm graph  | -w_size <br/> --start_date <br/> --end_date <br/> --save    |
|  get     | get the norm   | -w_size <br/> --start_date <br/> --end_date  |
|  crash_stats     |  get and plot statistics on crashs   | -w_size <br/> -year <br/> --test <br/> --plot </br> save  |
|  clean     | clean the hidden working database   |  |


#### Bottleneck
Access to the bottleneck of the persistence
```bash
pyconda manage.py bottleneck <\subcommand>
```
| subcommand     | explaination            | arguments   |
| :----------:   | :----------:             | :----------: |
|  visualise     | plot the bottleneck graph  | -w_size <br/> --start_date <br/> --end_date <br/> --save    |
|  get     | get the bottleneck   | -w_size <br/> --start_date <br/> --end_date  |
|  crash_stats     |  get and plot statistics on crashs   | -w_size <br/> -year <br/> --test <br/> --plot </br> save  |
|  clean     | clean the hidden working database   |  |


## Reference 
[1]: Marian Gidea and Yuri Katz. Topological data analysisof 
financial time series : Landscapes of crashes.PhysicaA 
: Statistical Mechanics and its Applications, 491 :820– 834, 2018 \
[2]: https://gudhi.inria.fr \
[3]: https://matplotlib.org


TODO ajouter Gudhi
