------------------

## AutoDC: An Automated Machine Learning Framework for Disease Classification.
AutoDC is a tailored AutoML system for targeting at different disease classification from gene expression data.
AutoDC is developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIM Lab</a> at Peking University.
The goal of AutoDC is to make machine learning easier to apply in biological data analysis.

Currently, AutoDC is compatible with: **Python >= 3.5**.

------------------

## Guiding principles

- __User friendliness.__ AutoDC needs few human assistance.

- __Easy extensibility.__ New ML algorithms are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making it suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

------------------

## Characteristics
- AutoDC supports AutoML on large dimensional biological datasets.

- AutoDC enables adaptive dimension reduction, transfer-learning, meta-learning and reinforcement learning techniques to make AutoML with more intelligent behaviors for biological data analysis.

------------------

## Example

Please check [examples](https://github.com/dingdian110/AutoDC/master/AutoDC_mdd_data_3600s.py).

------------------

## Installation

Before installing AutoDC, please install the necessary library [swig](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).

AutoDC requires SWIG (>= 3.0, <4.0) as a build dependency, and we suggest you to download & install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/).


Then, you can install AutoDC itself. There are two ways to install AutoDC:


#### Manual installation from the github source

```sh
git clone https://github.com/dingdian110/AutoDC.git && cd AutoDC
cat requirements.txt | xargs -n 1 -L 1 pip install
python setup.py install
```

### Tips on Installing Swig


- **for Arch Linux User:**

On Arch Linux (or any distribution with swig4 as default implementation), you need to confirm that the version of SWIG is in (>= 3.0, <4.0).

We suggest you to install [swig=3.0.12](https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/)..

```sh
./configure
make & make install
```
