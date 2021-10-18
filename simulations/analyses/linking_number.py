import json
import multiprocessing as mp
import os
import sys

import click
import numpy as np
import pandas as pd
import polychrom.polymer_analyses
import pyknotid.spacecurves
from polychrom.hdf5_format import list_URIs, load_URI


def error(msg):
    print(msg)


def count_intertwines(path, blocks, chain_length, exclude_particles, method):
    out = {block: None for block in blocks}
    blocks = np.array(list_URIs(path))[blocks]
    cen = chain_length // 2

    for idx, block in enumerate(blocks):
        coords = load_URI(block)["pos"][:-exclude_particles]
        left_arm = coords[:cen]
        right_arm = coords[cen:]

        if method == "pyknotid/closing_on_sphere":
            left_arm = pyknotid.spacecurves.SpaceCurve.closing_on_sphere(left_arm)
            right_arm = pyknotid.spacecurves.SpaceCurve.closing_on_sphere(right_arm)
        elif method == "manual_closing":
            np.insert(left_arm, 0, left_arm[0] * 10)
            np.insert(left_arm, -1, left_arm[-1] * 10)
            np.insert(right_arm, 0, right_arm[0] * 10)
            np.insert(right_arm, -1, right_arm[-1] * 10)
        elif method == "polychrom":
            linking_number = polychrom.polymer_analyses.getLinkingNumber(
                left_arm, right_arm
            )

        if method != "polychrom":
            linking_number = pyknotid.spacecurves.Link(
                lines=[left_arm, right_arm], verbose=False
            ).linking_number()

        out[blocks[idx]] = linking_number

    sim_name = path.split("/")[-2]
    return (sim_name, out)


@click.command()
@click.argument("in_dir", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "-b",
    "--block",
    "blocks",
    type=int,
    multiple=True,
    help="Simulation block to calculate intertwines from. Multiple uses allowed.",
)
@click.option(
    "--method",
    type=click.Choice(
        ["pyknotid", "pyknotid/closing_on_sphere", "manual_closing", "polychrom"],
        case_sensitive=False,
    ),
    default="pyknotid",
    help="The method to calculate the linking number. Default is 'pyknotid'.",
)
@click.option(
    "--chain-length",
    type=int,
    required=True,
    help="Length (not number of particles) of the chain in monomers.",
)
@click.option(
    "--exclude-particles",
    type=int,
    default=0,
    help="Number of particles to exclude from dataset.",
)
@click.option("--threads", type=int, help="Number of threads for parallel computation.")
def linking_number(
    in_dir,
    blocks,
    method,
    chain_length,
    exclude_particles,
    threads=mp.cpu_count(),
):
    """
    Counts the linking number between two arms of a chromosome from a polychrom simulation in the specified block files and outputs them to stdout.
    """
    out = {
        sim: None
        for sim in [
            d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))
        ]
    }

    def store(result):
        out[result[0]] = result[1]

    with mp.Pool(threads) as pool:
        results = []
        for name in out:
            sims_dir = os.path.join(in_dir, name)
            gap_p = int(name.split("gap_p=", 1)[1][0])
            chain_length = chain_length * (gap_p + 1)

            for sim in os.listdir(sims_dir):
                sim_path = os.path.join(sims_dir, sim)
                print(sim_path)
                result = pool.apply_async(
                    count_intertwines,
                    args=(
                        sim_path,
                        list(blocks),
                        chain_length,
                        exclude_particles,
                        method,
                    ),
                    callback=store,
                    error_callback=error
                )
                results.append(result)

        [result.wait() for result in results]

        pool.close()
        pool.join()

    # print(f"all processes finished: {results}")

    sys.stdout.write(json.dumps(out, indent=4))


if __name__ == "__main__":
    np.seterr(divide="ignore")
    linking_number()
