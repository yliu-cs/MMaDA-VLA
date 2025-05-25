import os
import autoroot
from rold.utils.misc import quiet
from argparse import ArgumentParser, Namespace
from rold.models.mmada import MMaDAConfig, MMaDAModelLM


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pretrained_mmada", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "MMaDA-8B-Base"))
    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    mmada_config = MMaDAConfig.from_pretrained(args.pretrained_mmada)
    print(f"{mmada_config=}")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)