from .processing import Processor
from .data import DataCollection
from .plotting import Plotter
from .aux_functions import *
from .consts import *

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(prog="IMBIE", description="IMBIE 2 Processor")
    parser.add_argument('-i', '--input-config', default=None, type=str, help="Input configuration file")
    parser.add_argument('-o', '--output-config', default=None, type=str, help="Output configuration file")
    parser.add_argument('-p', '--process-config', default=None, type=str, help="Processing configuration file")
    parser.add_argument('-g', '--graph-style', default=None, type=str, help="Graph rendering style configuration file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action="count", help="increase output verbosity")
    group.add_argument('-s', '--silent', action="store_true", help="do not print any output to the terminal window")

    args = parser.parse_args()

    proc_config = {}
    if args.process_config is not None:
        proc_config = load_json(args.process_config)
    in_config = {}
    if args.input_config is not None:
        in_config = load_json(args.input_config)
    out_config = {}
    if args.output_config is not None:
        out_config = load_json(args.output_config)
    plot_config = {}
    if args.graph_style is not None:
        plot_config = load_json(args.graph_style)


    if args.verbose >= 2:
        proc_config['verbose'] = True
    # create the plotter
    format = out_config.get('plot_format', None)
    output = out_config.get('output_path', None)
    plotter = Plotter(file_type=format, path=output, **plot_config)
    # create the DataCollection
    data = DataCollection(**in_config)
    # initialize the processor
    processor = Processor(data, **proc_config)

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
    if output is not None and out_config.get('output_files', False):
        processor.save(output)
        if not args.silent:
            print "saved output"

    # plot the graphs
    plots = out_config.get('plots', {})
    if plots.get('box_plots', False):
        processor.plot_boxes(plotter)
    if plots.get('dmdt_plots', False):
        processor.plot_dmdt(plotter)
    if plots.get('mass_plots', False):
        processor.plot_mass(plotter)
    if plots.get('dm_plots', False):
        processor.plot_cumulative(plotter)
    if not args.silent:
        print "completed plots"
