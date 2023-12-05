# Aty-TTS

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.


## Training

```
python train_aug.py -spk=<SPEAKER>
```

## Inference

```
python inference.py -f <TEXTFILE_PATH> -c <TTSMODEL_PATH> -o <OUTPUT_PATH> -v <VOCODER_PATH>
```

## Reference

[Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)