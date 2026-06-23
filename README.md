# CAMEC: Complexity-Aware Multi-Expert Collaboration for Reliable Chinese Medical QA
[![Paper](https://img.shields.io/badge/Paper-ACL%202026-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)]()

> **Accepted at ACL 2026 Main Conference**

CAMEC is a Chinese medical LLM framework combining hierarchical medical fine-tuning, complexity-aware expert routing, and multi-expert collaboration for reliable Chinese medical question answering.

This repository provides a system-level implementation of the framework proposed in:

> **CAMEC: Complexity-Aware Multi-Expert Collaboration for Reliable Chinese Medical Question Answering**  
> Yukang Wu, Xiyuan Jia, Jiayi Wu, Hongchen Yu, Yuhan Qiu, Guohua Wu  
> *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*

## Key Contributions

- **Complexity-aware adaptive collaboration**: a lightweight router predicts query complexity and selectively activates expert subsets (CoT / RAG / KG), reducing average expert invocations by **22.5%** while maintaining accuracy.
- **Multi-expert inference with judge-guided consensus**: three complementary experts reason in parallel; an LLM-as-a-Judge scores, refines, and synthesizes their outputs iteratively.
- **Hierarchical medical fine-tuning**: three-stage LoRA-based SFT (Domain Adaptation → Instruction Following → Clinical Reasoning) on Qwen3-8B.
- **State-of-the-art on four Chinese medical benchmarks**: 78.86% on CMExam, 84.15% on MedQA-CN, 78.51% on CMMLU-Med, 74.40% avg on CMB-exam — consistently outperforming HuatuoGPT-o1-7B across all benchmarks.

## Overview

Large language models have shown strong potential in medical question answering (QA), but remain unreliable in **clinically complex Chinese medical scenarios** due to hallucinations, weak factual grounding, and insufficient multi-perspective validation.

CAMEC addresses these challenges by introducing:

- **Hierarchical medical adaptation** via multi-stage LoRA-based supervised fine-tuning
- **Complexity-aware routing** to adaptively activate specialized experts
- **Parallel multi-expert reasoning** combining internal reasoning, retrieval, and structured knowledge
- **Judge-guided iterative refinement** for quality control and consensus synthesis

Rather than relying on a single reasoning path, CAMEC coordinates **Chain-of-Thought (CoT)**, **Retrieval-Augmented Generation (RAG)**, and **Knowledge Graph (KG)** experts under a unified **LLM-as-a-Judge** mechanism to produce medically correct, consistent, and interpretable outputs.

## Framework Overview

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>
<p align="center"><em>Overview of the CAMEC framework.</em></p>

## System Architecture

CAMEC consists of four core components:

### 1. Hierarchical Medical Fine-Tuning

The base model (**Qwen3-8B**) is adapted through a **three-stage LoRA-based SFT pipeline**:

- **Stage 1 — Domain Adaptation**: aligns the model with Chinese medical terminology and clinical language using Huatuo26M-Lite.
- **Stage 2 — Instruction Following**: improves multi-turn medical consultation and instruction adherence using HuatuoGPT2-SFT-GPT4-140K.
- **Stage 3 — Clinical Reasoning**: enhances diagnostic reasoning with chain-of-thought supervision from medical-o1-reasoning-SFT.

LoRA (rank 16) is applied with the base model frozen. A replay strategy mixes prior-stage samples into each subsequent stage to prevent catastrophic forgetting.

### 2. Complexity-Aware Expert Router

A lightweight **Qwen3-0.6B** router fine-tuned as a 3-way classifier predicts a **Medical Complexity Score (MCS)** based on symptom ambiguity, disease rarity, comorbidity risk, and diagnostic uncertainty. Queries are classified into three levels:

- **LOW** → CoT expert only
- **MEDIUM** → CoT + RAG
- **HIGH** → CoT + RAG + KG

This adaptive routing reduces average expert invocations from 3.00 to 2.32 (−22.5% cost) while retaining 99.8% of accuracy.

### 3. Multi-Expert Parallel Inference

For each query, activated experts generate structured reports **in parallel**:

- **CoT Expert (E_CoT)**: internal chain-of-thought reasoning over clinical patterns.
- **RAG Expert (E_RAG)**: dense retrieval over a Milvus vector index (Qwen3-Embedding-0.6B + huatuo_encyclopedia_qa) for factual grounding.
- **KG Expert (E_KG)**: structured reasoning over a Neo4j medical knowledge graph (~40k entities, ~290k relations) for consistency and safety.

### 4. Judge-Guided Iterative Evaluation

An LLM-as-a-Judge (same fine-tuned 8B model) evaluates each expert report on medical correctness (α=0.6), completeness (β=0.2), and safety (γ=0.2). Reports scoring below threshold τ=8 are iteratively refined (up to T_max=3 rounds). The judge synthesizes a final consensus via softmax-weighted fusion of high-confidence expert outputs.

## Main Results

### CMExam, MedQA-CN, CMMLU-Med (Table 1)

| Model | CMExam | MedQA-CN | CMMLU-Med |
|-------|-------:|----------:|----------:|
| Yi-1.5-9B | 68.1 | 75.8 | 64.2 |
| GLM-4-9B | 70.5 | 75.2 | 67.6 |
| Qwen2.5-7B | 70.4 | 71.4 | 70.5 |
| HuatuoGPT-II-7B | 67.4 | 73.7 | 58.4 |
| HuatuoGPT-o1-7B | 74.1 | 79.8 | 74.5 |
| **CAMEC (Ours)** | **78.86** | **84.15** | **78.51** |

CAMEC outperforms the previous state-of-the-art HuatuoGPT-o1-7B by +4.76, +4.35, and +4.01 points respectively.

### CMB-exam Sub-task Breakdown (Table 2)

| Model | Physician | Nurse | Pharmacist | Technician | Disciplines | Graduate | Avg |
|-------|----------:|------:|-----------:|-----------:|------------:|---------:|----:|
| *General Models* | | | | | | | |
| Qwen3-8B | 61.70 | 68.88 | 63.34 | 59.92 | 55.69 | 39.62 | 58.99 |
| GPT-4 | 59.90 | 69.31 | 52.19 | 61.50 | 59.69 | 54.19 | 59.46 |
| Yi-34B-Chat | 71.10 | 77.56 | 73.16 | **73.67** | 66.56 | 52.94 | 69.17 |
| *Medical Models* | | | | | | | |
| HuatuoGPT-II-7B | 64.55 | 63.75 | 64.04 | 62.67 | 63.08 | 54.31 | 62.07 |
| HuatuoGPT-II-13B | 67.85 | 66.12 | 66.19 | 64.06 | 65.40 | 59.62 | 64.87 |
| **CAMEC (Ours, 8B)** | **73.75** | **80.56** | **73.91** | 71.65 | **71.75** | **69.43** | **74.40** |

CAMEC (8B) surpasses Yi-34B-Chat (+5.23 avg) and HuatuoGPT-II-13B (+9.53 avg), demonstrating strong generalization across diverse medical subdomains.

### Router Efficiency (Table 5)

| Setting | Avg Experts | Accuracy (%) | Cost Reduction |
|---------|------------:|-------------:|---------------:|
| Full Experts (always 3) | 3.00 | 84.36 | — |
| **CAMEC with Router** | **2.32** | **84.15** | **−22.51%** |

The complexity-aware router reduces average expert invocations by 22.5% with negligible accuracy loss (−0.21%).

## Ablation Study

Results on CMExam (Table 3):

| Fine-Tune | CoT Expert | RAG Expert | KG Expert | Accuracy (%) |
|:---------:|:----------:|:----------:|:---------:|-------------:|
| — | — | — | — | 72.30 |
| ✓ | — | — | — | 73.91 |
| — | ✓ | ✓ | ✓ | 77.14 |
| ✓ | — | ✓ | ✓ | 77.05 |
| ✓ | ✓ | — | ✓ | 76.16 |
| ✓ | ✓ | ✓ | — | 76.83 |
| **✓** | **✓** | **✓** | **✓** | **78.86** |

Each component contributes positively; the full system achieves a +6.56 point improvement over the base model.

## Data

### SFT Training Data

| Stage | Dataset | Purpose |
|-------|---------|---------|
| Stage 1 | [Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite) | Domain adaptation |
| Stage 2 | [HuatuoGPT2-SFT-GPT4-140K](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K) | Instruction following |
| Stage 3 | [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | Clinical reasoning |

### RAG Corpus

- [huatuo_encyclopedia_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa) — large-scale Chinese medical QA corpus used to build the Milvus retrieval index.

### Knowledge Graph

- [RAGQnASystem](https://github.com/honeyandme/RAGQnASystem) — medical KG with ~40k entities and ~290k disease–symptom–treatment relations, loaded into Neo4j.

## Repository Structure

```
.
├── finetune/
│   ├── sft_train.sh          # Three-stage LoRA SFT pipeline
│   ├── lora_export.sh        # Export LoRA adapters
│   └── dataset/              # Medical SFT datasets
├── rag/
│   ├── import.py             # Import medical QA data into Milvus
│   ├── medical_qa.jsonl      # Example medical QA corpus
│   └── medical_qa.py         # RAG inference pipeline
├── judge/
│   └── judge.py              # Judge-guided evaluation and synthesis
├── router/
│   └── router.py             # Complexity-aware routing model
├── requirements.txt
└── README.md
```

## Installation

**Requirements**: Python 3.10+, CUDA 11.8+, one GPU with ≥24GB VRAM (experiments run on RTX 4090).

**Step 1 — Python environment**

```bash
conda create -n camec python=3.10
conda activate camec
pip install -r requirements.txt
```

**Step 2 — Start Milvus (vector retrieval service)**

```bash
# Requires Docker and Docker Compose
docker-compose up -d milvus
```

**Step 3 — Start Neo4j (knowledge graph)**

```bash
docker run -d \
  --name neo4j-camec \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

**Step 4 — Import RAG corpus and KG**

```bash
# Download huatuo_encyclopedia_qa and import into Milvus
python rag/import.py --data_path /path/to/huatuo_encyclopedia_qa

# Import medical knowledge graph into Neo4j
# (follow instructions in the RAGQnASystem repo linked above)
```

**Step 5 — Download or fine-tune the model**

To fine-tune from scratch using the three-stage SFT pipeline:

```bash
bash finetune/sft_train.sh    # runs all three stages sequentially
bash finetune/lora_export.sh  # merge LoRA adapters into the base model
```

## Quick Start

**1. Configure model paths**

```python
MODEL_NAME        = "path/to/finetuned-8B-model"   # fine-tuned Qwen3-8B
ROUTER_MODEL_PATH = "path/to/router-0.6B"           # fine-tuned Qwen3-0.6B router
EMB_PATH          = "path/to/Qwen3-Embedding-0.6B"
```

**2. Start external services**

```bash
# Milvus (vector retrieval)
docker run -d --name milvus -p 19530:19530 \
  -v $PWD/milvus_data:/var/lib/milvus milvusdb/milvus:v2.5.2 milvus run standalone

# Neo4j (knowledge graph)
docker run -d --name neo4j-camec -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

# Import RAG corpus
python rag/import.py
```

**3. Run inference**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from py2neo import Graph

# Load models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()

router_tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL_PATH)
router_model     = AutoModelForCausalLM.from_pretrained(ROUTER_MODEL_PATH, torch_dtype=torch.float16, device_map="auto").eval()

# Connect external knowledge sources
vecstore = Milvus(HuggingFaceEmbeddings(model_name=EMB_PATH), collection_name="huatuo_qa",
                  connection_args={"host": "localhost", "port": "19530"})
kg = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Example question (CMExam / MedQA-CN format)
example = {
    "Question": "患者男性，58岁，反复咳嗽、咳痰20年，加重伴喘息5年，最可能的诊断是？",
    "Options": [
        {"key": "A", "value": "支气管哮喘"},
        {"key": "B", "value": "慢性阻塞性肺疾病"},
        {"key": "C", "value": "支气管扩张"},
        {"key": "D", "value": "肺结核"},
    ]
}

# Step 1: Router predicts complexity (LOW / MEDIUM / HIGH)
complexity = assess_complexity(example)   # defined in router/router.py

# Step 2: Activate experts based on complexity
#   LOW   → CoT expert only
#   MEDIUM → CoT + RAG expert
#   HIGH   → CoT + RAG + KG expert
expert_answers, expert_texts = run_experts(example, complexity, model, tokenizer,
                                           vecstore=vecstore, kg=kg)

# Step 3: Judge scores reports and synthesizes final answer
final_answer = run_judge(example, expert_answers, expert_texts, model, tokenizer)
print(f"Predicted: {final_answer}")  # e.g. "B"
```

---

## Citation

If you use CAMEC in your research, please cite:

```bibtex
@inproceedings{wu2026camec,
  title     = {{CAMEC}: Complexity-Aware Multi-Expert Collaboration for Reliable
               Chinese Medical Question Answering},
  author    = {Wu, Yukang and Jia, Xiyuan and Wu, Jiayi and Yu, Hongchen and
               Qiu, Yuhan and Wu, Guohua},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (ACL 2026)},
  year      = {2026},
}
```

---

## Disclaimer

This repository is released for **research purposes only**. The system is not a certified medical device and must not be used for clinical diagnosis, treatment decisions, or patient care. Always consult a qualified healthcare professional for medical advice.
