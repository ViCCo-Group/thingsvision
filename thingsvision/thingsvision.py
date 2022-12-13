#!/usr/bin/env python3

""" This is the command-line interface for the thingsvision toolbox """

import argparse
import torch
import textwrap
import sys
import pkg_resources


def get_parsers():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
        prog="thingsvision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """ This is the thingsvision CLI. It allows you to extract features from images using state-of-the-art neural networks for Computer Vision."""
        ),
    )

    common_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    common_parser.add_argument(
        "--model-name",
        type=str,
        default="alexnet",
        help="Name of the model to use for feature extraction. (default: alexnet)",
    )
    common_parser.add_argument(
        "--source",
        type=str,
        default="torchvision",
        choices=["torchvision", "keras", "timm", "custom"],
        help="Source of the model to use for feature extraction. (default: torchvision)",
    )
    common_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for the extractor. (default: cpu)",
    )

    subparsers = parent_parser.add_subparsers(
        title="Subcommands",
        description="Valid subcommands are",
        help="Additional help available is available within  each subcommand",
        dest="command",
    )

    parser_model = subparsers.add_parser(
        "show-model",
        description="Shows the model architecture",
        parents=[common_parser],
    )

    parser_extract = subparsers.add_parser(
        "extract-features",
        description="Extract features from images",
        parents=[common_parser],
    )

    parser_extract.add_argument(
        "--image-root",
        type=str,
        help="Path to directory containing images. (default: ./images)",
        default="./images",
    )
    parser_extract.add_argument(
        "--class-names",
        type=str,
        help="optional list of class names for class dataset. (default: None)",
        default=None,
    )
    parser_extract.add_argument(
        "--file-names",
        type=str,
        help="optional list of file names for class dataset. (default: None)",
        default=None,
    )
    parser_extract.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for feature extraction. (default: 32)",
    )
    parser_extract.add_argument(
        "--out-path",
        type=str,
        default="./features",
        help="Path to directory where features should be stored. (default: ./features)",
    )
    parser_extract.add_argument(
        "--flatten-acts", action="store_true", help="Flatten activations before saving."
    )
    parser_extract.add_argument(
        "--model-parameters",
        type=str,
        default=None,
        help="Clip vision model has to be defined in the model_parameters dictionary. (default: None)",
    )
    parser_extract.add_argument(
        "--module-name",
        type=str,
        default="features.10",
        help="Name of the module to use for feature extraction. (default: features.10)",
    )
    parser_extract.add_argument(
        "--file-format",
        type=str,
        default="npy",
        help="File format to use for storing features. (default: npy)",
        choices=["npy", "hdf5", "mat", "txt"],
    )
    parent_parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    version = pkg_resources.get_distribution("thingsvision").version
    parent_parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {}".format(version),
        help="Show program's version number and exit.",
    )

    return parent_parser, parser_model, parser_extract


def main():
    parent_parser, parser_model, parser_extract = get_parsers()

    args = parent_parser.parse_args()
    if len(sys.argv) == 1:
        parent_parser.print_help(sys.stderr)
        sys.exit(1)
    elif len(sys.argv) == 2:
        print("\nPlease specify optional commands and arguments:\n")
        if args.command == "show-model":
            parser_model.print_help(sys.stderr)
        elif args.command == "extract-features":
            parser_extract.print_help(sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)

    from thingsvision import get_extractor
    from thingsvision.utils.storing import save_features
    from thingsvision.utils.data import ImageDataset, DataLoader

    extractor = get_extractor(
        model_name=args.model_name,
        model_path=None,
        pretrained=True,
        source=args.source,
        device=device,
    )

    if args.command == "show-model":
        extractor.show_model()
        sys.exit(1)

    if args.command == "extract-features":
        dataset = ImageDataset(
            root=args.image_root,
            out_path=args.out_path,
            backend=extractor.get_backend(),
            transforms=extractor.get_transformations(),
            class_names=args.class_names,
            file_names=args.file_names,
        )
        batches = DataLoader(
            dataset=dataset, batch_size=args.batch_size, backend=extractor.backend
        )

        features = extractor.extract_features(
            batches=batches,
            module_name=args.module_name,
            flatten_acts=args.flatten_acts,
        )

        save_features(
            features=features, out_path=args.out_path, file_format=args.file_format
        )


if __name__ == "__main__":
    main()
