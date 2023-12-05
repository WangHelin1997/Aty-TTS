# Aty-TTS
Official Pytorch Implementation of "[Improving fairness for spoken language understanding in atypical speech with Text-to-Speech](https://openreview.net/pdf?id=YU228ZUCOU)".

🔥Note that we will release the dataset (HeyJay) after checking the copyrights.
Instead, we provide all the source code and pre-trained models in this repo, so you can use the inference stage to generate atypical speech!

<img src="img\img.png" width="500">

## Abstract
Spoken language understanding (SLU) systems often exhibit suboptimal performance in processing atypical speech, typically caused by neurological conditions and motor impairments. Recent advancements in Text-to-Speech (TTS) synthesis-based augmentation for more fair SLU have struggled to accurately capture the unique vocal characteristics of atypical speakers, largely due to insufficient data. To address this issue, we present a novel data augmentation method for atypical speakers by finetuning a TTS model, called Aty-TTS. Aty-TTS models speaker and atypical characteristics via knowledge transferring from a voice conversion model. Then, we use the augmented data to train SLU models adapted to atypical speech. To train these data augmentation models and evaluate the resulting SLU systems, we have collected a new atypical speech dataset containing intent annotation. Both objective and subjective assessments validate that Aty-TTS is capable of generating high-quality atypical speech. Furthermore, it serves as an effective data augmentation strategy, contributing to more fair SLU systems that can better accommodate individuals with atypical speech patterns.

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd AtyTTS/model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.

## Data Preparation

Please download the data and pre-trained models from [Models](https://drive.google.com/drive/home), and then put them to the corresponding folders.


## Inference

```
python AtyTTS/inference.py -f <TEXTFILE_PATH> -c <TTSMODEL_PATH> -o <OUTPUT_PATH> -v <VOCODER_PATH>
```

An example of ```<TEXTFILE_PATH>``` can be found at ```AtyTTS/resources/text-to-syn.txt```.


## Training

You may need to change the paths to your own.

```
# Stage 1: Train the vocoder.
python AtyTTS/hifi-gan/train_aty.py
```

```
# Stage 2: Train the DuTa-VC model.
python DuTaVC/train_dec.py
```

```
# Stage 3: Train the Aty-TTS model.
python AtyTTS/train_aug.py --spk=<SPEAKER>
```


## References

If you find the code useful for your research, please consider citing:

```bibtex
@inproceedings{wang2023improving,
  title={Improving fairness for spoken language understanding in atypical speech with Text-to-Speech},
  author={Helin Wang and Venkatesh Ravichandran and Milind Rao and Becky Lammers and Myra Sydnor and Nicholas Maragakis and Ankur A. Butala and Jayne Zhang and Lora Clawson and Victoria Chovaz and Laureano Moro-Velazquez},
  booktitle={NeurIPS 2023 Workshop on Synthetic Data Generation with Generative AI},
  year={2023},
  url={https://openreview.net/forum?id=YU228ZUCOU}
}
```

```bibtex
@inproceedings{wang23qa_interspeech,
  author={Helin Wang and Thomas Thebaud and Jesús Villalba and Myra Sydnor and Becky Lammers and Najim Dehak and Laureano Moro-Velazquez},
  title={{DuTa-VC: A Duration-aware Typical-to-atypical Voice Conversion Approach with Diffusion Probabilistic Model}},
  year={2023},
  booktitle={Proc. INTERSPEECH 2023},
  pages={1548--1552},
  doi={10.21437/Interspeech.2023-2203}
}
```

This repo is inspired by:

1. [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)

2. [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC)
