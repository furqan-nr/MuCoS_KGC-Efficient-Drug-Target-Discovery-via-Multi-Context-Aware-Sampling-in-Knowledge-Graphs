import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the reproducible MuCoS tail-prediction pipeline end-to-end."
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing train/valid/test files.")
    parser.add_argument("--train-path", default=None, help="Optional explicit path to the training triples file.")
    parser.add_argument("--valid-path", default=None, help="Optional explicit path to the validation triples file.")
    parser.add_argument("--test-path", default=None, help="Optional explicit path to the test triples file.")
    parser.add_argument("--processed-dir", default=None, help="Directory for preprocessed contexts and vocabularies.")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs to run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training and evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate.")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum token length for context inputs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs.")
    parser.add_argument("--model-name", default=None, help="Hugging Face model name to fine-tune.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=None, help="Save a checkpoint every N optimizer steps.")
    parser.add_argument("--max-train-seconds", type=float, default=None, help="Stop training after this many seconds and save a checkpoint.")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume-training", dest="resume_training", action="store_true", help="Resume from latest_checkpoint.pth if available.")
    resume_group.add_argument("--no-resume-training", dest="resume_training", action="store_false", help="Start fresh even if a checkpoint exists.")
    parser.set_defaults(resume_training=None)
    return parser


def apply_environment_overrides(args):
    env_overrides = {
        "MUCOS_DATA_DIR": args.data_dir,
        "MUCOS_TRAIN_PATH": args.train_path,
        "MUCOS_VALID_PATH": args.valid_path,
        "MUCOS_TEST_PATH": args.test_path,
        "MUCOS_PROCESSED_DIR": args.processed_dir,
        "MUCOS_OUTPUT_DIR": args.output_dir,
        "MUCOS_CHECKPOINT_EVERY_STEPS": None if args.checkpoint_every_steps is None else str(args.checkpoint_every_steps),
        "MUCOS_MAX_TRAIN_SECONDS": None if args.max_train_seconds is None else str(args.max_train_seconds),
        "MUCOS_RESUME_TRAINING": None if args.resume_training is None else ("1" if args.resume_training else "0"),
    }
    for env_name, value in env_overrides.items():
        if value is not None:
            os.environ[env_name] = value


def resolve_dataset_paths(args):
    # If caller already set at least one explicit split path, require all explicit paths.
    explicit_count = sum(1 for p in [args.train_path, args.valid_path, args.test_path] if p)
    if explicit_count not in (0, 3):
        raise ValueError("Provide either all of --train-path/--valid-path/--test-path, or none of them.")

    if explicit_count == 3:
        return

    # No explicit split paths provided: try defaults first.
    data_dir = args.data_dir or "data"
    default_train = os.path.join(data_dir, "train.txt")
    default_valid = os.path.join(data_dir, "valid.txt")
    default_test = os.path.join(data_dir, "test.txt")

    if all(os.path.exists(p) for p in [default_train, default_valid, default_test]):
        return

    # Auto-detect extracted PharmKG-8k format (train.tsv/valid.tsv/test.tsv).
    pharm_dir = os.path.join(data_dir, "PharmKG-8k")
    pharm_train = os.path.join(pharm_dir, "train.tsv")
    pharm_valid = os.path.join(pharm_dir, "valid.tsv")
    pharm_test = os.path.join(pharm_dir, "test.tsv")

    if all(os.path.exists(p) for p in [pharm_train, pharm_valid, pharm_test]):
        os.environ["MUCOS_TRAIN_PATH"] = pharm_train
        os.environ["MUCOS_VALID_PATH"] = pharm_valid
        os.environ["MUCOS_TEST_PATH"] = pharm_test
        print(
            "[runner] Auto-detected PharmKG-8k TSV files in data/PharmKG-8k and will use them.",
            flush=True,
        )
        return

    raise FileNotFoundError(
        "Dataset files not found. Expected either data/train.txt, data/valid.txt, data/test.txt "
        "or data/PharmKG-8k/train.tsv, valid.tsv, test.tsv."
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        apply_environment_overrides(args)
        resolve_dataset_paths(args)
    except Exception as exc:
        parser.error(str(exc))

    import config_tail

    if args.epochs is not None:
        config_tail.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config_tail.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config_tail.LEARNING_RATE = args.learning_rate
    if args.max_length is not None:
        config_tail.MAX_LENGTH = args.max_length
    if args.seed is not None:
        config_tail.SEED = args.seed
    if args.model_name is not None:
        config_tail.MODEL_NAME = args.model_name

    import main_tail

    main_tail.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())