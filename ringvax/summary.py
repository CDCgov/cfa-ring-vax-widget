from typing import Optional, Sequence

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
def empirical_detection_prob(
    df: pl.DataFrame,
    detect_method: str,
    conditional_column: Optional[str] = None,
    not_: bool = False,
    numerator=False,
):
    """
    Computes the proportion of cases in `df` detected by method `detect_method` ("passive", "active", or "any" (for both)) without raising errors for 0/0 division.
    Can use `conditional_column` to compute either Pr(detect | condition met) (`numerator` == False) or Pr(detect and condition met) (`numerator` == True.)
    If `not_` == True, conditioning is on !`conditional_column`.

    Returns proportion, numerator count, and denominator count.
    """
    if conditional_column is not None:
        assert conditional_column in df.columns
        assert df.schema[conditional_column] == pl.Boolean

        if not_:
            df = df.with_columns(
                pl.col(conditional_column).not_().alias(conditional_column)
            )

        if not numerator:
            if df.filter(pl.col(conditional_column)).is_empty():
                return np.divide(0.0, 0.0)

            df = df.filter(pl.col(conditional_column))

    all_methods = ["passive", "active"]
    if detect_method == "any":
        match_methods = all_methods
    else:
        assert (
            detect_method in all_methods
        ), f"Unrecognized detection method {detect_method}"
        match_methods = [detect_method]

    detections = df.filter(pl.col("detect_method").is_in(match_methods))

    if numerator and conditional_column is not None:
        detections = detections.filter(pl.col(conditional_column))

    return (
        np.divide(detections.shape[0], df.shape[0]),
        detections.shape[0],
        df.shape[0],
    )


def summarize_detections(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get marginal detection probabilities from simulations.
    """
    n_infections = df.shape[0]

    # Add in eligibility conditions
    df = (
        df.join(
            df.select(["simulation", "id", "detected"]).rename({"id": "infector"}),
            on=["simulation", "infector"],
            how="left",
        )
        .unique(["simulation", "id"])
        .rename({"detected_right": "active_eligible"})
        .with_columns(
            is_index=pl.col("infector").is_null(),
            before_infectious=(pl.col("t_detected") < pl.col("t_infectious")),
        )
    )
    assert df.shape[0] == n_infections

    method = [
        "Either",
        "Either",
        "Either",
        "Either",
        "Passive",
        "Active",
    ]

    event = [
        "Detected",
        "Detected prior to infectiousness",
        "Detected",
        "Detected",
        "Detected",
        "Detected",
    ]

    among = [
        "All cases",
        "All cases",
        "Index cases",
        "Non-index cases",
        "Non-index cases",
        "Cases with detected infector",
    ]

    detect_info = [
        empirical_detection_prob(
            df,
            "any",
        ),
        empirical_detection_prob(df, "any", "before_infectious", numerator=True),
        empirical_detection_prob(
            df,
            "any",
            "is_index",
        ),
        empirical_detection_prob(
            df,
            "any",
            "is_index",
            not_=True,
        ),
        empirical_detection_prob(df, "passive", "is_index", not_=True),
        empirical_detection_prob(df, "active", "active_eligible"),
    ]
    return pl.DataFrame(
        {
            "Event": event,
            "Method": method,
            "Among": among,
            "Percent": [x[0] for x in detect_info],
            "Numerator": [x[1] for x in detect_info],
            "Denominator": [x[2] for x in detect_info],
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


def get_infection_counts_by_generation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get DataFrame of number of infections in each generation from simulations.
    """
    non_extinct = df.group_by("simulation", "generation").agg(num_infections=pl.len())

    gmax = int(max(df["generation"]))
    nsims = int(max(df["simulation"])) + 1

    all_extinct = [
        {"simulation": i, "generation": g, "num_infections": 0}
        for i in range(nsims)
        for g in range(gmax + 1)
    ]

    all_extinct = pl.DataFrame(all_extinct).cast(
        {"num_infections": pl.UInt32, "simulation": pl.Int32}
    )

    extinct = all_extinct.join(non_extinct, on=["simulation", "generation"], how="anti")

    return pl.concat([non_extinct, extinct])
