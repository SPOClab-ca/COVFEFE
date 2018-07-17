# HAS TO BE FIRST LINE
try:
    import matlab.engine
except ImportError:
    pass

import click

import os

from pipelines import pipeline_registry

from utils.logger import set_logger

@click.command()
@click.option("--in_folder", "-i", help="Path to input folder", required=True)
@click.option("--out_folder", "-o", help="Path to output folder", required=True)
@click.option("--pipeline", "-p", help="Pipeline to execute", required=True, type=click.Choice(sorted(list(pipeline_registry.all))))
@click.option("--num_threads", "-n", help="Number of concurrent threads to use for processing", type=click.INT, default=1)
@click.option("--log_file", "-l", help="File to log to", required=False)
def main(in_folder, out_folder, pipeline, num_threads, log_file=None):
    if not os.path.isdir(in_folder):
        print("Input folder %s does not exist" % in_folder)

    if not os.path.isdir(out_folder):
        print("Creating output folder %s" % out_folder)
        os.makedirs(out_folder)

    set_logger(log_file)

    p = pipeline_registry.all[pipeline](in_folder, out_folder, num_threads)
    p.run()


if __name__ == '__main__':
    main()