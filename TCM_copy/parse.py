import argparse
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser("Experiment For TGIKG")


    parser.add_argument("--wandb", default=False, action="store_true", help="Whether to adjust the parameters")
    parser.add_argument("--log_dir", default="logs/", type=str, help="Set the log path")

    parser.add_argument("--model_name", default="TCM", type=str, help="Set the model name")
    parser.add_argument("--version", default="tcdr", type=str, help="Set the version number")
    parser.add_argument("--seed", default=2025, type=int, help="Set the Seed")

    parser.add_argument("--test", default=False, action="store_true", help="Test the model")
    parser.add_argument("--train", default=False, action="store_true", help="Train the model")
    parser.add_argument("--predict", default=False, action="store_true", help="Single prediction")
    parser.add_argument("--print_model", default=False, action="store_true", help="print model")

    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--dim", default=64, type=int, help="model dimension")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--dp", default=0.1, type=float, help="dropout ratio")
    parser.add_argument("--epochs", default=3, type=int, help="the epochs for training")
    parser.add_argument("--early_stop", default=10, type=int, help="early stopping epochs")
    parser.add_argument("--freeze", type=int, help="Freeze the number of sessions of the training")
    parser.add_argument("--freeze_layers", default=[], help="Freeze Layers")

    parser.add_argument("--device", default=0, type=int, help="The device being used")
    parser.add_argument("--val_rate", default=0.2, type=float, help="Validation set ratio")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(PROJECT_ROOT, "dataset") + os.sep,
        type=str,
        help="Used to store the current TCM dataset",
    )
    parser.add_argument("--pre_train", type=str, default=None, help="The path of the pre-trained parameters")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(PROJECT_ROOT, "modules") + os.sep,
        type=str,
        help="model dir.",
    )

    args = parser.parse_args()
    return args
