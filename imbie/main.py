from .processing import Processor
from .data import DataCollection
from .plotting import Plotter
from .aux_functions import ask, ask_yes_no, parse_config
from .consts import *

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="IMBIE 2 Processor")
    parser.add_argument('-i', '--input', default=None, type=str, help="root directory from which to load input files")
    parser.add_argument('-o', '--output', default=None, type=str, help="output folder")
    parser.add_argument('-f', '--format', default=None, type=str, help="output graph format (png/svg/pdf/jpg)")
    parser.add_argument('-c', '--config', default=None, type=str, help="configuration file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action="count", help="increase output verbosity")
    group.add_argument('-s', '--silent', action="store_true", help="do not print any output to the terminal window")
    parser.add_argument('-a', '--ask', action="store_true", help="configure some parameters via command-line questions")

    args = parser.parse_args()

    config = {}
    if args.config is not None:
        config = parse_config(args.config)

    if args.verbose >= 2:
        config['verbose'] = True
    if args.ask:
        config.update({
            "grace_dmdt_method": ask("Select GRACE dm/dt method:", default="variable", variable=VARIABLE, fixed=FIXED),
            "reconciliation_method": ask("Select reconciliation method:", default="x4", x3=X3, x4=X4),
            "random_walk": ask_yes_no("Use random walk?")
        })
    # create the plotter
    conf = {
        # 'figure.figsize': (10, 10),
        # 'figure.dpi': 160
    }

    plotter = Plotter(file_type=args.format, path=args.output, **conf)
    # create the DataCollection
    data = DataCollection(args.input)
    # initialize the processor
    processor = Processor(data, **config)

    # load & assign the data
    processor.assign()
    if not args.silent:
        print "loaded all data"
    # merge & accumulate
    processor.merge()
    if not args.silent:
        print "merged data sources"

    # save the results
    if args.verbose >= 1:
        processor.print_stats()
    if args.output is not None:
        processor.save(args.output)
        if not args.silent:
            print "saved output"

    # plot the graphs
    if not args.ask or ask_yes_no("Show box plots?"):
        processor.plot_boxes(plotter)
    if not args.ask or ask_yes_no("Plot dm/dt?"):
        processor.plot_dmdt(plotter)
    if not args.ask or ask_yes_no("Plot mass?", default="no"):
        processor.plot_mass(plotter)
    if not args.ask or ask_yes_no("Plot cumulative mass?"):
        processor.plot_cumulative(plotter)
    if not args.silent:
        print "completed plots"
