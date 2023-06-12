# MetaBind
MetaBind is a meta-learning model for bioactivity prediction. Traditional ML/DL models are trained based on point-wised aggregated data which neglect the assay heterogeneity. Instead of aggregated data pointwisely, the MetaBind model learns the assay heterogeneity and bioactivity simultaneously and shows drastic improvement in bioactivity prediction across diverse protein targets and assay types compared to conventional baselines.

## Requirements
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [rdkit](https://www.rdkit.org/)
- [biopython](https://biopython.org/)
- [pytorch](https://pytorch.org/)
- [torch-scatter](https://pypi.org/project/torch-scatter/)
- [dgl](https://www.dgl.ai/)
 
## Installation
pip install .

## Usage
cd script <br/>
python prepare\_data.py # prepare data set and pre-generate ligand graph and sequence feature tensor <br/>
python train\_model.py --model aggregated --split chronological --datatype all --nepoch 50 --batchsize 256 --data ../data/Aggregated/chronological_train.txt    # This train aggregated model <br/> 
python evaluate\_model.py --model aggregated --split chronological --datatype all --parameter ../parameters/aggregated_model_chronological.pth # Model evaluation <br/>

## Citing MetaBind
If you have used MetaBind in the course of your research, please cite our preprint.

To cite the preprint, please use this bibtex entry:

'''\
@article{chan-2023,\
    title = {Embracing assay heterogeneity with neural processes for markedly improved bioactivity predictions},\
    author = {Lucian Chan and Marcel Verdonk and Carl Poelking},\
    year = {2023}\
}\
'''
