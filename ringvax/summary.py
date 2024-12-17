from collections import Counter
from typing import Container, Sequence

import numpy as np
import polars as pl

from ringvax import Simulation

infection_schema = {
    "infector": pl.String,
    "generation": pl.Int64,
    "t_exposed": pl.Float64,
    "t_infectious": pl.Float64,
    "t_recovered": pl.Float64,
    "infection_rate": pl.Float64,
    "detected": pl.Boolean,
    "detect_method": pl.String,
    "t_detected": pl.Float64,
    "infection_times": pl.List(pl.Float64),
}
"""
An infection as a polars schema
"""


def infection_to_predf(infection: dict) -> dict:
    """
    Convert an infection to a dictionary which can be fed to polars.DataFrame()
    """
    dfable = {}
    for k, v in infection.items():
        if isinstance(v, np.ndarray):
            assert k == "infection_times"
            dfable |= {k: [[float(vv) for vv in v]]}
        else:
            assert isinstance(v, str) or not isinstance(v, Container)
            dfable |= {k: [v]}
    return dfable


def get_all_person_properties(sims: Sequence[Simulation]) -> pl.DataFrame:
    """
    Get a dataframe of all properties of all infections
    """
    g_max = [sim.params["n_generations"] for sim in sims]
    assert (
        len(Counter(g_max).items()) == 1
    ), "Aggregating simulations with different `n_generations` is nonsensical"

    i_max = [sim.params["max_infections"] for sim in sims]
    assert (
        len(Counter(i_max).items()) == 1
    ), "Aggregating simulations with different `max_infections` is nonsensical"

    per_sim = []
    for idx, sim in enumerate(sims):
        g_max.append(sim.params["n_generations"])
        i_max.append(sim.params["max_infections"])
        per_sim.append(
            pl.concat(
                [
                    (
                        pl.DataFrame(
                            infection_to_predf(infection) | {"simulation": [idx]}
                        ).cast(infection_schema)  # type: ignore
                    )
                    for infection in sim.infections.values()
                ]
            )
        )
    return pl.concat(per_sim)


def summarize_detections(df: pl.DataFrame) -> pl.DataFrame:
    nsims = len(df["simulation"].unique())
    n_infections = df.shape[0]
    n_active_eligible = n_infections - nsims
    detection_counts = df.select(pl.col("detect_method").value_counts()).unnest(
        "detect_method"
    )

    count_nodetect = 0
    if detection_counts.filter(pl.col("detect_method").is_null()).shape[0] == 1:
        count_nodetect = detection_counts.filter(pl.col("detect_method").is_null())[
            "count"
        ]
    count_active, count_passive = 0, 0
    if detection_counts.filter(pl.col("detect_method") == "active").shape[0] == 1:
        count_active = detection_counts.filter(pl.col("detect_method") == "active")[
            "count"
        ]
    if detection_counts.filter(pl.col("detect_method") == "passive").shape[0] == 1:
        count_passive = detection_counts.filter(pl.col("detect_method") == "passive")[
            "count"
        ]

    return pl.DataFrame(
        {
            "prob_detect": 1.0 - count_nodetect / n_infections,
            "prob_active": count_active / n_active_eligible,
            "prob_passive": count_passive / n_infections,
            "prob_detect_before_infectious": df.filter(pl.col("detected"))
            .filter(pl.col("t_detected") < pl.col("t_infectious"))
            .shape[0]
            / n_infections,
        }
    )


def summarize_infections(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        n_infections=pl.col("infection_times").list.len(),
        t_noninfectious=pl.min_horizontal(
            [pl.col("t_detected"), pl.col("t_recovered")]
        ),
    ).with_columns(
        duration_infectious=(pl.col("t_noninfectious") - pl.col("t_infectious"))
    )

    return pl.DataFrame(
        {
            "mean_infectious_duration": df["duration_infectious"].mean(),
            "sd_infectious_duration": df["duration_infectious"].std(),
            # This is R_e
            "mean_n_infections": df["n_infections"].mean(),
            "sd_n_infections": df["n_infections"].std(),
        }
    )


def prob_control_by_gen(df: pl.DataFrame, gen: int) -> float:
    g_max = df.group_by("simulation").agg(pl.col("generation").max())
    return (g_max.filter(pl.col("generation") < gen).shape[0]) / (g_max.shape[0])


def get_outbreak_size_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("simulation")
        # length of anything in the grouped dataframe is number of infections
        .agg(pl.col("t_exposed").len().alias("size"))
    )
