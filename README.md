# Signal-AI

The purpose of Signal-AI is to simplify operating with signals and to apply
deep-learning models on signal data. 
Currently supported AI types are

- Time Series Classification (Signal Classification)
- Signal Decomposition
- Noise Reduction
- Event List Generation
- Signal Autoencoding

with the option of using custom PyTorch models added to `src/signalai/models`.

## Installation
```
python setup.py develop
```

## Usage 

### Signal

The Signal class allows to easily manipulate with signal data. The basic operations are
```python
import signalai as sai
m0 = sai.read_audio('data/example/dataset/Actions_-_Devils_Words_-_1.aac')  # sai.Signal class
m1 = sai.read_audio('data/example/dataset/Actions_-_One_Minute_Smile_-_3.aac')

m0_crop = m0.crop([5*44100, 10*44100])  # cropping by samples, 44100 represents the sampling frequency
m0_crop.play()  # play the audio

numpy_array = m0_crop.data_arr  # 2D matrix, axis 0 and 1 are the channel and time axes, respectively 
```

### Train your own model

Signal-AI is based on [Taskchain](https://pypi.org/project/taskchain/) which keeps 
order in machine learning projects by splitting parts of code into individual
tasks. This also allows you to put all the dataset and network setting into
YAML config files. See some example configs in `configs` folder.

In order to train your own model, create a config file based on the example configs.
The model can be trained using the code 

```python
from signalai import config
from taskchain.task import Config

config_path = config.CONFIGS_DIR / 'models' / 'example' / 'augment' / 'sepnet.yaml'
conf = Config(
    config.TASKS_DIR,  # where Taskchain data (including the model itself) should be stored
    config_path,
    global_vars=config,  # set global variables
)
chain = conf.chain()
chain.set_log_level('CRITICAL')

signal_model = chain.trained_model.value
```

After this, the `signal_model` variable contains trained model of class 
`signalai.torch_core.TorchSignalModel`. TaskChain makes sure the model is saved
based on the config setting. Any changes to the config lead to the model retraining.

### Future

The future development of this project will try to include some autoencoders,
more time-frequency transformations as well as more data augmentations.


### Reference

Example dataset is taken from

Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, and R. Bittner, The MUSDB18 corpus
for music separation, Dec. 2017. doi: 10.5281/zenodo.1117372. [Online]. Available:
https://doi.org/10.5281/zenodo.1117372.
