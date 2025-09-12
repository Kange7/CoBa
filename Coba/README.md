# Contextual boundary-aware network for semantic segmentation of complex land transportation point cloud scenes

###  Setup
This code has been tested with Python 3.8, Tensorflow 2.4, CUDA 11.2 and .

- Setup python environment
```
conda create -n coba python=3.8
source activate coba
pip install -r helper_requirements.txt
sh compile_op.sh
```


###  Train & Test

```
python main_$data_you_want_use$.py --mode train & test
```


