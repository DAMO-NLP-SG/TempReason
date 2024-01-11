# TempReason
Data and implementation for "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models"

Due to the size limit of Github repositories, please download the dataset by the following commands:
```
git lfs install
git clone https://huggingface.co/datasets/tonytan48/TempReason
```
You may put the data in your desired directory, we suggest putting them under 'data'.

Backbone model Training for T5-SFT
```
bash tempreason_train.sh
```


Further training with TSRL 
```
cd trlx
python examples/tsqa/run_t5_qa.py
```

Pretrained Checkpoints
```
Accessible in this Google Drive folder: https://drive.google.com/drive/folders/1Pezx__jBraBKLMFqWnDVOQUhMV1eFoLv?usp=sharing
```

If you find our work useful, please cite our paper as:
```bibtex
@inproceedings{tan-etal-2023-towards,
    title = "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models",
    author = "Tan, Qingyu  and
      Ng, Hwee Tou  and
      Bing, Lidong",
    booktitle = "Proceedings of ACL",
    year = "2023",
    url = "https://arxiv.org/abs/2306.08952",

}
```
