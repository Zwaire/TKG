import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data.api_text_generator import QwenAPIGenerator, SimpleTextGenerator
from data.preprocessor import ICEWS14Preprocessor
from modules.clustering.online_clustering import OnlineClustering
from modules.data_module import TemporalKGDataset
from modules.embedding import CompleteEmbeddingModule
from modules.trainer.temporal_trainer import TemporalKGTrainer
from utils.logger import ExperimentLogger


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_processed_data(config, logger):
    data_dir = Path(config["data"]["data_path"])
    processed_dir = data_dir / "processed"
    required_files = [
        processed_dir / "train_triples.npy",
        processed_dir / "valid_triples.npy",
        processed_dir / "test_triples.npy",
        processed_dir / "entity2id.json",
        processed_dir / "relation2id.json",
    ]

    if all(p.exists() for p in required_files):
        logger.info("Processed data found. Skip preprocessing.")
        return processed_dir

    logger.info("Processed data not found. Running preprocessing...")
    preprocessor = ICEWS14Preprocessor(str(data_dir))
    num_entities, num_relations = preprocessor.load_mappings()
    train = preprocessor.load_triples("train.txt")
    valid = preprocessor.load_triples("valid.txt")
    test = preprocessor.load_triples("test.txt")
    preprocessor.save_processed_data(str(processed_dir), train, valid, test, logger)
    logger.info(
        f"Preprocessing done. num_entities={num_entities}, num_relations={num_relations}"
    )
    return processed_dir


def sync_entity_relation_counts(config, processed_dir: Path):
    with open(processed_dir / "entity2id.json", "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(processed_dir / "relation2id.json", "r", encoding="utf-8") as f:
        relation2id = json.load(f)

    config.setdefault("model", {})["num_entities"] = len(entity2id)
    config["model"]["num_relations"] = len(relation2id)
    return entity2id, relation2id


def generate_text_cache(config, logger, split: str, use_api: bool):
    data_dir = Path(config["data"]["data_path"])
    processed_dir = data_dir / "processed"
    triples_path = processed_dir / f"{split}_triples.npy"
    if not triples_path.exists():
        raise FileNotFoundError(f"Missing triples file: {triples_path}")

    triples = np.load(triples_path)
    with open(processed_dir / "entity2id.json", "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(processed_dir / "relation2id.json", "r", encoding="utf-8") as f:
        relation2id = json.load(f)
    id2entity = {int(v): k for k, v in entity2id.items()}
    id2relation = {int(v): k for k, v in relation2id.items()}

    cache_dir = Path(config["data"]["text_generation"].get("cache_dir", "./data/text_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if use_api:
        logger.info("Using API text generator (Qwen compatible endpoint).")
        generator = QwenAPIGenerator(cache_dir=str(cache_dir))
    else:
        logger.info("Using local simple text generator.")
        generator = SimpleTextGenerator(cache_dir=str(cache_dir))

    triples_list = [(int(s), int(p), int(o), int(t)) for s, p, o, t in triples]
    texts = generator.batch_generate(triples_list, id2entity, id2relation)

    out_file = cache_dir / f"{split}_texts.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(texts)} texts -> {out_file}")


def build_trainer(config):
    device = config["training"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    config["training"]["device"] = device

    data_dir = Path(config["data"]["data_path"])
    text_cache_dir = Path(config["data"]["text_generation"].get("cache_dir", "./data/text_cache"))
    processed_dir = data_dir / "processed"

    train_dataset = TemporalKGDataset(
        data_path=str(processed_dir),
        history_window=config["data"].get("history_window", 5),
        time_granularity=config["data"].get("time_granularity", "day"),
        split="train",
        text_cache_dir=str(text_cache_dir),
    )
    valid_dataset = TemporalKGDataset(
        data_path=str(processed_dir),
        history_window=config["data"].get("history_window", 5),
        time_granularity=config["data"].get("time_granularity", "day"),
        split="valid",
        text_cache_dir=str(text_cache_dir),
    )

    model = CompleteEmbeddingModule(
        num_entities=config["model"]["num_entities"],
        num_relations=config["model"]["num_relations"],
        config=config["model"]["embedding"],
    )

    cluster_cfg = config["model"].get("clustering", {})
    clustering_module = OnlineClustering(
        n_clusters=cluster_cfg.get("n_clusters", 20),
        update_frequency=cluster_cfg.get("update_frequency", 10),
        clustering_method=cluster_cfg.get("method", "minibatch_kmeans"),
        use_gpu=(device == "cuda"),
    )

    trainer = TemporalKGTrainer(
        model=model,
        clustering_module=clustering_module,
        config=config,
        device=device,
    )
    return trainer, train_dataset, valid_dataset


def main():
    parser = argparse.ArgumentParser(description="Temporal KG training pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["preprocess", "generate_texts", "train", "resume"],
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--use_api", action="store_true", help="Use Qwen API for text generation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="temporal_training")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs in config")
    parser.add_argument("--max_samples", type=int, default=None, help="Max temporal samples per epoch (for debug/smoke)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    if args.epochs is not None:
        config["training"]["num_epochs"] = int(args.epochs)
    if args.max_samples is not None:
        config["training"]["max_samples_per_epoch"] = int(args.max_samples)

    exp_logger = ExperimentLogger(
        experiment_name=args.experiment,
        log_dir=project_root / "experiments",
        config=config,
    )
    logger = exp_logger.logger

    processed_dir = ensure_processed_data(config, logger)
    sync_entity_relation_counts(config, processed_dir)

    if args.mode == "preprocess":
        logger.info("Preprocess finished.")
    elif args.mode == "generate_texts":
        generate_text_cache(config, logger, split=args.split, use_api=args.use_api)
    else:
        trainer, train_dataset, valid_dataset = build_trainer(config)
        exp_logger.log_model_info(trainer.model)

        if args.mode == "resume":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required when mode=resume")
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        trainer.train_temporal_sequence(train_dataset=train_dataset, valid_dataset=valid_dataset)
        logger.info("Training finished.")

    exp_logger.finish()


if __name__ == "__main__":
    main()