# ecg_encoder

This is a research-oriented, open-source repository for full-stack, pretraining, finetuning, and evaluation of ECG-specific neural networks. We provide the infrastructure to conduct experiments on all publicly available ECG: [https://github.com/ELM-Research/ecg_preprocess](https://github.com/ELM-Research/ecg_preprocess). Please prepare the ECG datasets before considering this repository.

## Installation

1. `git clone https://github.com/ELM-Research/ecg_encoder.git`

2. `cd` into repo and `uv sync`.

## Scripts

Please view the `scripts` folder for examples on how to pretrain and finetune. We have 


## Acknowledgements <a name="ack"></a>
This work is done in collaboration with the Mario Lemieux Center for Heart Rhythm Care at Allegheny General Hospital.

We thank Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao, Hyoeun Kang, Wenhao Ding, Haohong Lin, Shiqi Liu, Xiaoyu (Simon) Song, Tony Chen, Atharva Mhaskar, Zhepeng Cen, Yihang Yao, and Dylan Leong for their helpful discussions, feedbacks, and support in developing ECG-Bench.

We thank the authors of [ECG-Byte](https://github.com/willxxy/ECG-Byte), [MERL](https://github.com/cheliu-computation/MERL-ICML2024), [ST-MEM](https://github.com/bakqui/ST-MEM), [ECG-QA](https://github.com/Jwoo5/ecg-qa), [ECG-Chat](https://github.com/YubaoZhao/ECG-Chat), [PULSE](https://github.com/AIMedLab/PULSE), and [GEM](https://github.com/lanxiang1017/GEM) for their code and publicly released datasets.

Lastly, we thank [HuggingFace](https://huggingface.co/) for providing the APIs for the models.

## License <a name="license"></a>

This repository contains code licensed under the MIT License, except for the following `.py` files in the `ecg_bench/models/encoder` directory: `st_mem.py`, `mlae.py`, `mtae.py`. These files are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please view the original license in their [respective repository](https://github.com/bakqui/ST-MEM?tab=License-1-ov-file#readme) for more details.