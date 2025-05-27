import os
import argparse

from auxiliary_script_funcs import get_ratings


parser = argparse.ArgumentParser()

parser.add_argument("--source-files",nargs="*", type=str, default=[])

parser.add_argument("--style", type=bool, default=False)
parser.add_argument("--len", type=bool, default=False)
parser.add_argument("--no-control", type=bool, default=True)
parser.add_argument("--pen-only-model", type=bool, default=False)

parser.add_argument("--mean-scores", type=bool, default=False)
parser.add_argument("--elo", type=bool, default=False)

parser.add_argument("--unite-battles", type=bool, default=False)
parser.add_argument("--battles-file", type=str, default='')

parser.add_argument("--load-results", type=bool, default=False)
parser.add_argument("--results-file", type=str, default='')

parser.add_argument("--print-results", type=bool, default=False)

parser.add_argument("--model", type=str, default='')

args = parser.parse_args()

get_ratings(
    source_files=args.source_files,
    model=args.model,
    save_all_battles=args.unite_battles,
    newfilename=args.battles_file,
    answerfile=args.results_file,
    load_results=args.load_results,
    print_results=args.print_results,
    style_control=args.style,
    len_control=args.len,
    no_control=args.no_control,
    pen_only_model=args.pen_only_model,
    get_mean_scores=args.mean_scores,
    get_elo_ratings=args.elo
)