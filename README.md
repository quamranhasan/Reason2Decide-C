# Reason2Decide-C
This repository contains the code and processed datasets accompanying our work on Reason2Decide-C (https://www.mdpi.com/2073-431X/15/5/279)

## Abstract

Large Language Models (LLMs) used for clinical decision support must not only make accurate predictions but also generate rationales that are consistent with, and sufficient for, those predictions. Building on Reason2Decide, a two-stage rationale-driven multi-task framework, we propose Reason2Decide-C (R2D-C, where C denotes cycle consistency), which augments Reason2Decide’s stage 2 training with confidence-adaptive scheduled sampling and cycle-consistent rationale-to-label training. In stage 1, we pretrain our model on rationale generation. In stage 2, we jointly train on label prediction and rationale generation, gradually replacing gold labels with model-predicted labels based on confidence. Simultaneously, we feed the rationale logits back into the model to recover the label, thus enforcing explanation sufficiency. We evaluate R2D-C on one proprietary triage dataset, as well as public biomedical QA and reasoning datasets. Across model sizes, R2D-C substantially improves rationale–prediction consistency (where stage 1 and stage 2 predictions agree) and sufficiency (where the rationale alone recovers the ground-truth label) over other baselines while matching or modestly improving predictive performance (F1); in several settings, R2D-C surpasses 40× larger foundation models. Ablations confirm that the full combination is optimal, maximizing alignment and LLM-as-a-Judge rationale quality. These results demonstrate that confidence-adaptive scheduled sampling and cycle-consistent rationale-to-label training substantially enhance explanation alignment without sacrificing accuracy.

---
 
## Datasets
The repository contains pre-processed datasets with train/validation/test splits 
* PubMedQA
* DDXPlus
* MedReason

The clinical triage dataset used in the paper is not publicly available and therefore cannot be redistributed.

---

## Code Overview


### `src/`

Contains code for Reason2Decide-C training and inference:

`r2dc_stage1.py`: Rationale Generation Training

`r2dc_stage1.py`: Joint Training for both rationale and prediction

`infer.py`: evaluation script

---

## Excluded Components

### DSS

The **distilling-step-by-step (DSS)**  and  **Reason2Decide (R2D)**  components are intentionally **not included** in this repository.

Reason:

* In our experiments, both are **direct modifications of the original implementation**
* The original DSS and R2D source codes are publicly available.

Please refer to the original authors’ repository for the DSS implementation. Can be found here: https://github.com/google-research/distilling-step-by-step
Please refer to the original authors’ repository for the R2D implementation. Can be found here: https://github.com/quamranhasan/Reason2Decide

---

## Installation

We recommend using a Conda environment.
```bash
conda create -n reason2decide_c python=3.10
conda activate reason2decide_cs
pip install -r requirements.txt
```

---


## Training Script Usage:

For r2dc_stage1 the following can be used:
```bash
torchrun --nproc_per_node=4 src/r2dc_stage1.py \
    --train_file PATH_TO_TRAIN_FILE \
    --valid_file PATH_TO_VALID_FILE \
    --batch_size BATCH_SIZE \
    --grad_steps GRADIENT_ACCUMULATION_STEPS \
    --eval_steps EVAL_EVERY_N_STEPS \
    --max_steps MAX_TRAINING_STEPS \
    --model_name PRETRAINED_MODEL_NAME \
    --output_dir OUTPUT_DIR \
    --seed RANDOM_SEED \
    --max_input_length MAX_INPUT_TOKENS

```
For r2dc_stage2, the following can be used:
```bash
torchrun --nproc_per_node=NUM_GPUS src/r2dc_stage2.py \
    --train_file PATH_TO_TRAIN_FILE \
    --valid_file PATH_TO_VALID_FILE \
    --batch_size BATCH_SIZE \
    --grad_steps GRADIENT_ACCUMULATION_STEPS \
    --eval_steps EVAL_EVERY_N_STEPS \
    --max_steps MAX_TRAINING_STEPS \
    --seed RANDOM_SEED \
    --stage1_model_dir PATH_TO_STAGE1_CHECKPOINT \
    --output_dir OUTPUT_DIR \
    --max_input_length MAX_INPUT_TOKENS
```

---

## Citation
If you find this repository useful, please consider citing:

```bash

@Article{computers15050279,
AUTHOR = {Hasan, H M Quamran and Babiker, Housam Khalifa Bashier and Kim, Mi-Young and Goebel, Randy},
TITLE = {Reason2Decide-C: Adaptive Cycle-Consistent Training for Clinical Rationales},
JOURNAL = {Computers},
VOLUME = {15},
YEAR = {2026},
NUMBER = {5},
ARTICLE-NUMBER = {279},
URL = {https://www.mdpi.com/2073-431X/15/5/279},
ISSN = {2073-431X},
ABSTRACT = {Large Language Models (LLMs) used for clinical decision support must not only make accurate predictions but also generate rationales that are consistent with, and sufficient for, those predictions. Building on Reason2Decide, a two-stage rationale-driven multi-task framework, we propose Reason2Decide-C (R2D-C, where C denotes cycle consistency), which augments Reason2Decide’s stage 2 training with confidence-adaptive scheduled sampling and cycle-consistent rationale-to-label training. In stage 1, we pretrain our model on rationale generation. In stage 2, we jointlytrain on label prediction and rationale generation, gradually replacing gold labels with model-predicted labels based on confidence. Simultaneously, we feed the rationale logits back into the model to recover the label, thus enforcing explanation sufficiency. We evaluate R2D-C on one proprietary triage dataset, as well as public biomedical QA and reasoning datasets. Across model sizes, R2D-C substantially improves rationale–prediction consistency (where stage 1 and stage 2 predictions agree) and sufficiency (where the rationale alone recovers the ground-truth label) over other baselines while matching or modestly improving predictive performance (F1); in several settings R2D-C surpasses 40× larger foundation models. Ablations confirm that the full combination is optimal, maximizing alignment and LLM-as-a-Judge rationale quality. These results demonstrate that confidence-adaptive scheduled sampling and cycle-consistent rationale-to-label training substantially enhance explanation alignment without sacrificing accuracy.},
DOI = {10.3390/computers15050279}
}

```
---

## Dataset References

```bash
@inproceedings{jin-etal-2019-pubmedqa,
    title = "{P}ub{M}ed{QA}: A Dataset for Biomedical Research Question Answering",
    author = "Jin, Qiao  and
      Dhingra, Bhuwan  and
      Liu, Zhengping  and
      Cohen, William  and
      Lu, Xinghua",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1259/",
    doi = "10.18653/v1/D19-1259",
    pages = "2567--2577",
    abstract = "We introduce PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts. The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA instances. Each PubMedQA instance is composed of (1) a question which is either an existing research article title or derived from one, (2) a context which is the corresponding abstract without its conclusion, (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and (4) a yes/no/maybe answer which summarizes the conclusion. PubMedQA is the first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is required to answer the questions. Our best performing model, multi-phase fine-tuning of BioBERT with long answer bag-of-word statistics as additional supervision, achieves 68.1{\%} accuracy, compared to single human performance of 78.0{\%} accuracy and majority-baseline of 55.2{\%} accuracy, leaving much room for improvement. PubMedQA is publicly available at \url{https://pubmedqa.github.io}."
}
```

```bash
@misc{wu2025medreasonelicitingfactualmedical,
      title={MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs}, 
      author={Juncheng Wu and Wenlong Deng and Xingxuan Li and Sheng Liu and Taomian Mi and Yifan Peng and Ziyang Xu and Yi Liu and Hyunjin Cho and Chang-In Choi and Yihan Cao and Hui Ren and Xiang Li and Xiaoxiao Li and Yuyin Zhou},
      year={2025},
      eprint={2504.00993},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.00993}, 
}
```

```bash
@misc{tchango2022ddxplusnewdatasetautomatic,
      title={DDXPlus: A New Dataset For Automatic Medical Diagnosis}, 
      author={Arsene Fansi Tchango and Rishab Goel and Zhi Wen and Julien Martel and Joumana Ghosn},
      year={2022},
      eprint={2205.09148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2205.09148}, 
}
```
