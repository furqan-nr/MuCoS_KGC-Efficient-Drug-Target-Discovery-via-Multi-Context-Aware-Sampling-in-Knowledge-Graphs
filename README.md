# MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs

**Official PyTorch implementation** of the paper:

> **MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs**  
> Haji Gul, Abdul Ghani Naim, Ajaz Ahmad Bhat  
> School of Digital Science, Universiti Brunei Darussalam  
> **BioNLP at ACL 2025**  
> Proceedings of the 24th Workshop on Biomedical Language Processing

**Paper Link**: [https://aclanthology.org/search/?q=MuCos-KGC](https://aclanthology.org/2025.bionlp-1.27/)


## Installation  
Clone the repository and install the dependencies:  

```bash
git clone <your-repository-url>
cd MuCoS
pip install -r requirements.txt
```


### Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```bash
venv\Scripts\activate
pip install -r requirements.txt
```


## Requirements  
Typical dependencies include:  

- torch
- transformers
- numpy
- scikit-learn
- pandas
- tqdm

Install them with:
```bash
pip install torch transformers numpy scikit-learn pandas tqdm
```


## Data Preparation  
Prepare your biomedical knowledge graph data in the format expected by the training scripts.  
A typical dataset directory may look like this:   

```bash
data/
├── train.txt
├── valid.txt
└── test.txt
```


Each line should represent a triple in the form:

```bash
head_entity    relation    tail_entity
```

Example:

```bash
DrugA    targets    ProteinX
DrugB    treats     DiseaseY
```

If your code uses a different folder structure or configuration file, update the paths in config.py accordingly.


## How to Run  
1. Move into the training module
   
```bash
cd relation_prediction
```

2. Train the model

```bash
python train.py
```


This will start the MuCoS training pipeline for relation prediction.

3. Configure model and dataset settings  
Before running training, review and adjust the settings in:  

```bash
relation_prediction/config.py
```

### Tail Prediction (Reproducible)

The reproducible tail-prediction pipeline precomputes contexts and trains once, selecting the best checkpoint by validation MRR.

1. Place data files in a folder such as `data/` or set environment variables:

```bash
set MUCOS_DATA_DIR=path\to\data
```

2. Run the tail-prediction pipeline:

```bash
python Tail_Prediction/main_tail.py
```

Outputs are written to `processed/` and `outputs/` by default. You can override them with:

```bash
set MUCOS_PROCESSED_DIR=path\to\processed
set MUCOS_OUTPUT_DIR=path\to\outputs
```

#### Typical parameters to configure include:  
- dataset path
- pretrained language model
- batch size
- learning rate
- number of epochs
- sampling parameters n and k
- output directory


## Output  
Depending on your training configuration, the code may produce:  

- trained model checkpoints
- prediction outputs
- evaluation logs
- MRR and ranking metrics
- saved experiment artifacts

It is recommended to store outputs in folders such as:  

```bash
outputs/
checkpoints/
logs/
results/
```

If these are not yet created in the code, you may add them for cleaner experiment management.  



---

## Overview

MuCoS is a knowledge-graph-based framework for **drug-target interaction prediction**. It models drug-target discovery as a **link prediction task** over heterogeneous biomedical knowledge graphs containing drugs, proteins, diseases, pathways, and related biomedical entities.

Traditional knowledge graph embedding methods often rely on expensive **negative sampling** and may generalize poorly to unseen drug-target pairs. MuCoS addresses these limitations through a **Multi-Context-Aware Sampling** strategy that selectively focuses on highly informative structural contexts and combines them with **BERT-based textual representations**.

By integrating **structural** and **contextual** information, MuCoS reduces computational overhead while improving prediction quality for both general relation prediction and drug-target discovery.

---

## Abstract

Accurate prediction of drug-target interactions is critical for accelerating drug discovery. In this work, we frame drug-target prediction as a link prediction task on heterogeneous biomedical knowledge graphs (KGs) that integrate drugs, proteins, diseases, pathways, and other relevant entities. Conventional KG embedding methods such as TransE and ComplExSE are hindered by their reliance on computationally intensive negative sampling and their limited generalization to unseen drug-target pairs.

To address these challenges, we propose **Multi-Context-Aware Sampling (MuCoS)**, a novel framework that prioritizes high-density neighbours to capture salient structural patterns and integrates these with contextual embeddings derived from BERT. By unifying structural and textual modalities and selectively sampling highly informative patterns, MuCoS eliminates the need for negative sampling, significantly reduces computational overhead, and improves generalization to unseen drug-target pairs and targets.

Extensive experiments on the **KEGG50k** and **PharmKG-8k** datasets demonstrate that MuCoS outperforms competitive baselines, achieving:

- **up to 13% improvement in MRR** for general relation prediction on **KEGG50k**
- **up to 22% improvement in MRR** on **PharmKG-8k**
- **up to 6% improvement** in dedicated **drug-target relation prediction** on **KEGG50k**

---

## Key Contributions

- **Multi-Context-Aware Sampling** for biomedical knowledge graphs
- **No negative sampling** required during training
- **BERT-based contextual modeling** for structured biomedical triples
- Improved generalization to **unseen drug-target pairs**
- Reduced computational complexity through **density-based sampling**
- Support for both **relation prediction** and **tail prediction**

---

## Features

- **Density-based multi-context sampling**
  - Head context
  - Tail context
  - Relation context

- **Transformer-based sequence classification**
  - `bert-base-uncased`
  - DistilBERT
  - RoBERTa

- **Efficient sampling**
  - `n`: number of top-density neighbours used for head/tail contexts
  - `k`: number of top-density entity pairs used for relation context

- **Reduced complexity**
  - Sampling reduces complexity from `O(avg_density + avg_appearance)` to `O(2n + k)`

- **Dual prediction tasks**
  - **Relation prediction**: predict the missing relation in `(h, ?, t)`
  - **Tail prediction**: predict the missing entity in `(h, r, ?)`

- **Two evaluation settings**
  - **General setting**: all relations and entities in the KG
  - **Drug-target-specific setting**: only drug-target interactions

---


##  Model Pipeline

### Figure 1: MuCoS Overall Pipeline
![Figure 1: MuCoS Pipeline](model1.png)  
**Figure 1:** A concise overview of the MuCoS model pipeline, which is designed to predict general and drug-target relations and tail entities. The boxes on the left show the input sequence to the BERT model, where (h) head, (Hc) head context, (t) tail, (Tc) tail context, (r) relation, and (Rc) relation context. This integrated context is passed through the BERT model with a linear classifier and softmax function to generate probabilities for relations and tail.


### Figure 2: Head Context (Hc) Construction with Sampling
![Figure 2: Head Context Sampling](hc.png)  
**Figure 2:** MuCoS $\mathcal{H}_c$ construction. The left graphical view illustrates one hop head $h$ context, which consists of the set of relations $\mathcal{R}(h)$ ($r_1, r_2, r_3, r_4, r_5, r_6$) and the set of neighbouring tail entities $\mathcal{E}(h)$ ($e_1, e_2, e_3, e_4, e_5, e_6$) associated with the head entity $h$. The middle view shows the sampling process, where only the top-$n$ (suppose $n = 3$) tail entities $e$ are selected and concatenated (||) based on their density $\rho(e)$, to calculate the optimized head context $\mathcal{H}_c$.

### Figure 3: Relation Context (Rc) Construction with Sampling
![Figure 3: Relation Context Sampling](Rc_n.png)  
**Figure 3:** $\mathcal{R}_c$ construction. The left view illustrates the relationship $r_1$ and entities (head, tail) connected by $r_1$. The graph in the middle depicts optimization, selecting the top $k$ (suppose $k = 2$) entities based on density $\rho$, retaining pairs such as $(e_2, e_3)$ and $(e_6, e_7)$. The optimized context $\mathcal{R}_c$ is aggregated using concatenation ($||$), as shown in the right section.


## Project Structure

```bash
MuCoS/
├── relation_prediction/          # General relation prediction: (h, ?, t)
│   ├── config.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── model.py
│   └── train.py
├── model1.png
├── hc.png
├── Rc_n.png
├── requirements.txt
├── README.md
└──
```


### Tasks Supported

1. Relation Prediction  
Predict the missing relation in:

```bash
(h, ?, t)
```

2. Tail Prediction  
Predict the missing tail entity in:
```bash
(h, r, ?)
```

3. Drug-Target Prediction

Specialized evaluation focused on drug-target interaction discovery.



## Experimental Settings  

MuCoS supports two main evaluation settings:   

- General Setting: Uses all entities and relations in the biomedical knowledge graph.
- 
- Drug-Target-Specific Setting: Restricts evaluation to drug-target interaction pairs for focused biomedical discovery.  





## Notes

- The method combines graph structure with textual context  
- It is especially useful when negative sampling is expensive or unstable  
- Larger pretrained models may require more GPU memory and training time   
- Sampling parameters n and k should be tuned based on the dataset and task



## How to Cite  
If you use this repository, code, or findings in your research, please cite:

```bash
@inproceedings{gul-etal-2025-mucos,  
    title = "MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs",  
    author = "Gul, Haji  and Naim, Abdul Ghani  and Bhat, Ajaz Ahmad",  
    editor = "Demner-Fushman, Dina  and Ananiadou, Sophia  and Miwa, Makoto  and Tsujii, Junichi",  
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing at Association for Computational Linguistics (ACL)",  
    month = aug,  
    year = "2025",  
    address = "Viena, Austria",  
    publisher = "Association for Computational Linguistics (ACL)",  
    url = "https://aclanthology.org/2025.bionlp-1.27/",  
    doi = "10.18653/v1/2025.bionlp-1.27",  
    pages = "319--327",  
    ISBN = "979-8-89176-275-6"  
}
```

## Acknowledgment

This work was carried out at the School of Digital Science, Universiti Brunei Darussalam.



