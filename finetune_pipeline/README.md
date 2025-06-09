# Fine-tuning Pipeline for Scientific Document Retrieval

This directory provides a unified pipeline for fine-tuning three state-of-the-art retrieval models—**DPR**, **SPECTER2**, and **SPLADE**—on custom scientific datasets. Each model is organized in its own subdirectory, with modular code for training, evaluation, and model management.

## Supported Models

- **DPR (Dense Passage Retrieval)**
  - [facebook/dpr-question_encoder-single-nq-base](https://huggingface.co/facebook/dpr-question_encoder-single-nq-base)
  - Dense dual-encoder architecture for open-domain question answering and retrieval.

- **SPECTER2**
  - [allenai/specter2](https://huggingface.co/allenai/specter2)
  - Transformer-based model for scientific document representation, leveraging citation-informed contrastive learning.

- **SPLADE (Sparse Lexical and Expansion Model)**
  - [naver/splade-v3](https://huggingface.co/naver/splade-v3)
  - Sparse neural retrieval model combining lexical and expansion-based representations for efficient and effective search.

## Directory Structure

- `dpr/`, `specter2/`, `splade/`
  - Each contains a `finetune.py` script for model-specific fine-tuning.
  - Additional modules (e.g., `model.py`, `trainer.py`) define model architectures and training logic.

- `base/`
  - Shared components for all models:
    - `data.py`: Dataset and data loader definitions (e.g., triplet datasets for retrieval).
    - `loss.py`: Loss functions (e.g., triplet margin loss).
    - `retrieval.py`: Abstract retrieval classes and utilities.

## Fine-tuning Workflow

Each model’s `finetune.py` script follows a similar workflow:
1. **Load Data**: Reads triplet-formatted training data (query, positive, negative).
2. **Initialize Model**: Loads the pre-trained model from HuggingFace.
3. **Set Up Trainer**: Prepares the training loop, optimizer, and evaluation logic.
4. **Train and Validate**: Fine-tunes the model and saves the best checkpoint.

See each model’s directory for detailed configuration and usage.

## References

- [DPR Paper](https://arxiv.org/abs/2004.04906)
- [SPECTER2 Paper](https://arxiv.org/abs/2305.14722)
- [SPLADE Paper](https://arxiv.org/abs/2104.06967)

---

**Note:** For more details on each model’s fine-tuning process, refer to the respective `finetune.py` and supporting modules in each subdirectory.
