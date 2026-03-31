import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ptflops import get_model_complexity_info

from config import get_config
from logger import create_logger
from models import build_model, build_mtl_model
from utils import load_checkpoint


MACS_PER_GMAC = 1e9
PARAMS_PER_MILLION = 1e6


def parse_option():
    parser = argparse.ArgumentParser("DiTASK FLOPs summary", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", required=True, help="checkpoint to inspect")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--output-dir", type=str, help="explicit directory to save summary artifacts")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--name", type=str, help="override model name")
    parser.add_argument("--tag", help="experiment tag")
    parser.add_argument("--tasks", type=str, default="depth", help="comma separated tasks")
    parser.add_argument("--nyud", type=str, help="NYUD dataset path")
    parser.add_argument("--pascal", type=str, help="PASCAL dataset path")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=0, help="local rank")
    parser.add_argument(
        "--print-per-layer-stat",
        action="store_true",
        help="print per-layer ptflops statistics",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable ptflops verbose output",
    )
    args = parser.parse_args()
    args.resume = args.checkpoint
    config = get_config(args)
    return args, config


def set_random_seed(seed, deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        cudnn.benchmark = True


def get_output_dir(args):
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    return os.path.join(checkpoint_dir, "standalone_flops")


def normalize_image_size(config):
    image_size = config.DATA.IMG_SIZE
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError(f"Expected DATA.IMG_SIZE to have length 2, got {image_size}.")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def build_model_for_flops(config, device):
    backbone = build_model(config)
    model = build_mtl_model(backbone, config) if config.MTL else backbone
    model.to(device)
    return model


def load_model_for_flops(config, checkpoint_path, device, logger):
    model = build_model_for_flops(config, device)
    load_config = config.clone()
    load_config.defrost()
    load_config.MODEL.RESUME = checkpoint_path
    load_config.EVAL_MODE = True
    load_config.freeze()
    load_checkpoint(load_config, model, None, None, None, logger)
    return model


def compute_flops_summary(model, input_shape, print_per_layer_stat=False, verbose=False):
    model.eval()
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=bool(print_per_layer_stat),
        verbose=bool(verbose),
    )
    macs = float(macs)
    params = float(params)
    return {
        "input_shape": list(input_shape),
        "macs": macs,
        "gmacs": macs / MACS_PER_GMAC,
        "gflops": (2.0 * macs) / MACS_PER_GMAC,
        "parameter_count": int(params),
        "parameter_millions": params / PARAMS_PER_MILLION,
    }


def build_summary_payload(args, config, flops_summary):
    return {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_name": config.MODEL.NAME,
        "tasks": list(config.TASKS),
        "input_shape": flops_summary["input_shape"],
        "macs": flops_summary["macs"],
        "gmacs": flops_summary["gmacs"],
        "gflops": flops_summary["gflops"],
        "parameter_count": flops_summary["parameter_count"],
        "parameter_millions": flops_summary["parameter_millions"],
        "notes": {
            "gflops_definition": "Computed as 2 * MACs / 1e9.",
            "device": "cpu",
        },
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args, config = parse_option()
    seed = int(args.seed if args.seed is not None else config.SEED)
    set_random_seed(seed, bool(args.deterministic))

    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="flops_summary")

    device = torch.device("cpu")
    model = load_model_for_flops(config, args.checkpoint, device, logger)

    input_h, input_w = normalize_image_size(config)
    flops_summary = compute_flops_summary(
        model,
        (3, input_h, input_w),
        print_per_layer_stat=bool(args.print_per_layer_stat),
        verbose=bool(args.verbose),
    )
    payload = build_summary_payload(args, config, flops_summary)

    save_json(os.path.join(output_dir, "flops_summary.json"), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
