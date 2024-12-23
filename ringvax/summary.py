from typing import Sequence

import numpy as np
import polars as pl

from ringvax import Simulation

infection_schema = pl.Schema(
    {
        "id": pl.String,
        "infector": pl.String,
        "infectees": pl.List(pl.String),
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
)
"""
An infection as a polars schema
"""

assert set(infection_schema.keys()) == Simulation.PROPERTIES


def get_all_person_properties(
    sims: Sequence[Simulation], exclude_termination_if: list[str] = ["max_infections"]
) -> pl.DataFrame:
    """
    Get a dataframe of all properties of all infections
    """
    assert (
        len(set(sim.params["n_generations"] for sim in sims)) == 1
    ), "Aggregating simulations with different `n_generations` is nonsensical"

    assert (
        len(set(sim.params["max_infections"] for sim in sims)) == 1
    ), "Aggregating simulations with different `max_infections` is nonsensical"

    return pl.concat(
        [
            _get_person_properties(sim).with_columns(simulation=sim_idx)
            for sim_idx, sim in enumerate(sims)
            if sim.termination not in exclude_termination_if
        ]
    )


def _get_person_properties(sim: Simulation) -> pl.DataFrame:
    """Get a DataFrame of all properties of all infections in a simulation"""
    return pl.from_dicts(
        [_prepare_for_df(x) for x in sim.infections.values()], schema=infection_schema
    )


def _prepare_for_df(infection: dict) -> dict:
    """
    Convert numpy arrays in a dictionary to lists, for DataFrame compatibility
    """
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in infection.items()
    }


@np.errstate(invalid="ignore")
def summarize_detections(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get marginal detection probabilities from simulations.
    """
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
            "prob_detect": 1.0 - np.divide(count_nodetect, n_infections),
            "prob_active": np.divide(count_active, n_active_eligible),
            "prob_passive": np.divide(count_passive, n_infections),
            "prob_detect_before_infectious": np.divide(
                df.filter(pl.col("detected"))
                .filter(pl.col("t_detected") < pl.col("t_infectious"))
                .shape[0],
                n_infections,
            ),
        }
    )


def summarize_infections(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get summaries of infectiousness from simulations.
    """
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
    """
    Compute the probability of control in generation (probability extinct in or before this generation) for all simulations
    """
    n_sim = df["simulation"].unique().len()
    size_at_gen = (
        df.with_columns(
            pl.col("generation") + 1,
            n_infections=pl.col("infection_times").list.len(),
        )
        .with_columns(size=pl.sum("n_infections").over("simulation", "generation"))
        .unique(subset=["simulation", "generation"])
        .filter(
            pl.col("generation") == gen,
            pl.col("size") > 0,
        )
    )
    return 1.0 - (size_at_gen.shape[0] / n_sim)


def get_total_infection_count_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get DataFrame of all total outbreak sizes from simulations
    """
    return (
        df.group_by("simulation")
        # length of anything in the grouped dataframe is number of infections
        .agg(pl.col("t_exposed").len().alias("size"))
    )
