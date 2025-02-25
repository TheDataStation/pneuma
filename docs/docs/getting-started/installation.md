To install the latest stable release from PyPI:

```bash
$ pip install pneuma
```

To install the most recent version from the repository:

```bash
$ git clone https://github.com/TheDataStation/Pneuma.git
$ cd Pneuma
$ pip install -r requirements.txt
```

### Installation Note

To ensure smooth installation and usage, we **strongly recommend** installing `Miniconda` (follow [this](https://docs.anaconda.com/miniconda/install/)). Then, create a new environment and install the CUDA Toolkit:

```bash
$ conda create --name pneuma python=3.12.2 -y
$ conda activate pneuma
$ conda install -c nvidia cuda-toolkit -y
```
