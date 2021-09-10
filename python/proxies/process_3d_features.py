import argparse, os
import subprocess
import tempfile
import logging

import numpy as np

import proxies.errors as errors


def parse_feature_file(f_in):
    with open(f_in, "r") as f:
        feature_list = []
        for l in f.read().splitlines():
            feature_list.append([float(e) for e in l.split()])
    return np.array(feature_list)


def process(
    f_in,
    f_out,
    technique,
    axis_cuts=0,
    grid_size=0,
    num_points=500,
):

    temp = tempfile.NamedTemporaryFile()

    # Badness
    cmd = [
        os.path.dirname(os.path.abspath(__file__))
        + "/../src/pcl_feature "
        + "{} ".format(f_in)
        + "{} ".format(temp.name)
        + "{} ".format(technique)
        + "-p {} ".format(num_points)
    ]

    if grid_size != 0:
        cmd[0] += "-g {} ".format(grid_size)

    if axis_cuts != 0:
        cmd[0] += "-c {} ".format(axis_cuts)

    logging.debug("Executing command in the shell: \n{}".format(cmd))
    # Call the pcl script
    app = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    # Filter and display errors, if they occur
    for line in app.stdout:
        if ("error" in line) or ("Error" in line) or ("not" in line):
            raise errors.PCLError()

    np.savez(f_out, features=parse_feature_file(temp.name))


def process_SHOT(
    f_in,
    f_out,
    grid_size=None,
    axis_cuts=None,
):
    return process(
        f_in,
        f_out,
        "SHOT",
        grid_size=0,
        num_points=500,
    )


def process_PFH(
    f_in,
    f_out,
    grid_size=None,
    axis_cuts=None,
):
    return process(
        f_in,
        f_out,
        "PFH",
        grid_size=0,
        num_points=500,
    )


def process_global_SHOT(
    f_in,
    f_out,
):
    return process(
        f_in,
        f_out,
        "SHOT",
        grid_size=1,
        axis_cuts=0,
        num_points=1000,
    )


def process_global_PFH(
    f_in,
    f_out,
):
    return process(
        f_in,
        f_out,
        "PFH",
        grid_size=1,
        axis_cuts=0,
        num_points=1000,
    )


def process_global_grid_offset_SHOT(
    f_in,
    f_out,
    grid_size=1,
    axis_cuts=None,
):
    return process(
        f_in,
        f_out,
        "SHOT",
        grid_size=grid_size,
        axis_cuts=0,
        num_points=1000,
    )


def process_global_grid_offset_PFH(
    f_in,
    f_out,
    grid_size=1,
    axis_cuts=None,
):
    return process(
        f_in,
        f_out,
        "PFH",
        grid_size=grid_size,
        axis_cuts=0,
        num_points=1000,
    )


def process_global_axis_cuts_SHOT(
    f_in,
    f_out,
    grid_size=None,
    axis_cuts=1,
):
    return process(
        f_in,
        f_out,
        "SHOT",
        grid_size=0,
        axis_cuts=axis_cuts,
        num_points=1000,
    )


def process_global_axis_cuts_PFH(
    f_in,
    f_out,
    grid_size=None,
    axis_cuts=1,
):
    return process(
        f_in,
        f_out,
        "PFH",
        grid_size=0,
        axis_cuts=axis_cuts,
        num_points=1000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates 3D features from a given input object. Object "
        + "must have vertices and vertex normals defined. This is really just "
        + "a wrapper for pcl code."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="The obj file to process. Must contain vertices and vertex normals.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="The feature file to generate (txt).",
    )
    parser.add_argument(
        "--technique",
        "-t",
        type=str,
        required=True,
        help="Which feature to commpute [PFH, SHOT]",
    )
    parser.add_argument(
        "--global",
        "-g",
        dest="_global",
        default=False,
        action="store_true",
        help="If passed, will compute feature globally.",
    )
    parser.add_argument(
        "--grid_size",
        "-s",
        type=int,
        default=1,
        help="The grid_size with which to compute global features",
    )
    args = parser.parse_args()

    assert args.technique in ["PFH", "SHOT"], "Unknown feature: {}.".format(
        args.technique
    )

    # Process
    val = process(
        f_in=args.input,
        f_out=args.output,
        technique=args.technique,
        global_feat=args._global,
        grid_size=args.grid_size,
    )
