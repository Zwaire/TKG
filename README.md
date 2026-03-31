# TKG Task

This project implements temporal KG prediction with the following pipeline:

1. Structural representation: Temporal RGCN
2. Semantic representation: BERT/Sentence-Transformer text encoder
3. Joint optimization: reconstruction + cross-modal contrastive + temporal consistency + clustering regularization
4. Future entity prediction

## Project Structure

- `main.py`: unified entry
- `config.yaml`: project config
- `data/`: raw data, processed triples, and text cache
- `models/`: temporal RGCN encoder
- `modules/`: dataset, embedding, clustering, trainer
- `utils/`: logging and helper utilities

## Quick Start

```bash
pip install -r requirements.txt
python main.py --mode preprocess
python main.py --mode train
```

## Modes

- `preprocess`: build `train/valid/test_triples.npy` and id mappings
- `generate_texts`: generate text cache for a split
- `train`: run training
- `resume`: resume from checkpoint

Examples:

```bash
python main.py --mode generate_texts --split train
python main.py --mode train --experiment temporal_training
python main.py --mode train --epochs 5
python main.py --mode resume --checkpoint checkpoints/checkpoint_epoch5.pt
```

## Config

Use root `config.yaml`.

## Notes

- Data files are preserved during refactor.
- Checkpoints are written to `checkpoints/` by default.
