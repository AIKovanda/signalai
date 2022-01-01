# Signal-AI

The purpose of Signal-AI is to simplify operating with signals and to apply
deep-learning models on signal data. 
Currently supported AI types are

- Time Series Classification (Signal Classification)
- Signal decomposition
- Noise reduction
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
m0 = sai.read_audio('path/to/song0.mp3')  # sai.Signal class
m1 = sai.read_audio('path/to/song1.mp3')

m0_crop = m0.crop([5*44100, 10*44100])  # cropping by samples, 44100 represents the sampling frequency
m0_crop.play()  # play the audio

m0_crop.show()  # shows the amplitude

numpy_array = m0_crop.data_arr  # 2D matrix, axis 0 and 1 are the channel and time axes, respectively 
```
### Training a model
Signal-AI is based on [Taskchain](https://pypi.org/project/taskchain/) which keeps 
order in machine learning projects by splitting parts of code into individual
tasks. This also allows to put all the dataset and network setting into YAML config
files. See some example configs in `configs` folder. 

```
python inference.py --input path/to/an/audio/file
```

### Example

Some example configurations including training on custom data. Note that the example
data only contains three songs 10 seconds long. All these configs are set to train
on 30 batches which is too little for any good model.

```bash
python scripts/decomposer.py --model_config example/augment2d/timexception_selu_at_magpha.yaml --eval_dir data/example/predict
python scripts/decomposer.py --model_config example/augment2d/timexception_selu_noat_mag.yaml --eval_dir data/example/predict

python scripts/decomposer.py --model_config example/augment/decomposer1L255_nores_bot64_n64.yaml --eval_dir data/example/predict
python scripts/decomposer.py --model_config example/augment/se_simple_noat_sep.yaml --eval_dir data/example/predict
```

### Train your own model

In order to train your own model, create a config file based on the example configs.
The model can be trained using the code 

```python
from signalai import config
from taskchain.task import Config

config_path = config.CONFIGS_DIR / 'models' / 'example' / 'my_model.yaml'
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