from typing import Sequence

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

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
    n_sims = len(df["simulation"].unique())
    n_infections = df.shape[0]

    parent_detected_df = (
        df.join(
            df.select(["simulation", "id", "detected"]).rename({"id": "infector"}),
            on=["simulation", "infector"],
            how="left",
        )
        .unique(["simulation", "id"])
        .rename({"detected_right": "infector_detected"})
    )
    assert_frame_equal(
        parent_detected_df.drop("infector_detected", "infectees", "infection_times"),
        df.drop("infectees", "infection_times"),
        check_row_order=False,
    )
    parent_detected_counts = parent_detected_df["infector_detected"].value_counts()
    assert (
        parent_detected_counts.filter(pl.col("infector_detected").is_null())["count"][0]
        == n_sims
    )
    n_active_eligible = parent_detected_counts.filter(pl.col("infector_detected"))[
        "count"
    ][0]

    all_detection_counts = df.select(pl.col("detect_method").value_counts()).unnest(
        "detect_method"
    )

    nonindex_detection_counts = (
        df.filter(pl.col("infector").is_not_null())
        .select(pl.col("detect_method").value_counts())
        .unnest("detect_method")
    )

    index_detection_counts = (
        df.filter(pl.col("infector").is_null())
        .select(pl.col("detect_method").value_counts())
        .unnest("detect_method")
    )

    count_nodetect = 0
    if all_detection_counts.filter(pl.col("detect_method").is_null()).shape[0] == 1:
        count_nodetect = all_detection_counts.filter(pl.col("detect_method").is_null())[
            "count"
        ][0]

    count_active, count_passive_nonindex = 0, 0
    if (
        nonindex_detection_counts.filter(pl.col("detect_method") == "active").shape[0]
        == 1
    ):
        count_active = nonindex_detection_counts.filter(
            pl.col("detect_method") == "active"
        )["count"][0]
    if (
        nonindex_detection_counts.filter(pl.col("detect_method") == "passive").shape[0]
        == 1
    ):
        count_passive_nonindex = nonindex_detection_counts.filter(
            pl.col("detect_method") == "passive"
        )["count"][0]

    count_index_not = 0
    if not index_detection_counts.filter(pl.col("detect_method").is_null()).is_empty():
        count_index_not = index_detection_counts.filter(
            pl.col("detect_method").is_null()
        )["count"][0]

    detect_types = [
        "Any",
        "Index case",
        "Active (among eligible)",
        "Passive (among non-index cases)",
        "Before infectiousness",
    ]

    detect_probs = [
        1.0 - np.divide(count_nodetect, n_infections),
        1.0 - np.divide(count_index_not, n_sims),
        np.divide(count_active, n_active_eligible),
        np.divide(count_passive_nonindex, n_infections - n_sims),
        np.divide(
            df.filter(pl.col("detected"))
            .filter(pl.col("t_detected") < pl.col("t_infectious"))
            .shape[0],
            n_infections,
        ),
    ]
    return pl.DataFrame(
        {
            "Detection category": detect_types,
            "Probability": detect_probs,
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
