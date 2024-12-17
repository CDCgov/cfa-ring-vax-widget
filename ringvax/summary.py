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

    count_nodetect = detection_counts.filter(pl.col("detect_method").is_null())["count"]
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


# def summarize_generations(df: pl.DataFrame, g_max: int) -> pl.DataFrame:
#     gens = np.arange(g_max)
#     n_extinct_by = np.array([
#         (df[f"generation_{g}"] == 0).sum()
#         for g in range(g_max)
#     ])
#     n_sims_applicable = df.shape[0] - n_extinct_by
#     pr_extinct = n_extinct_by / len(sims)
#     mean_size_at = np.array([df[f"generation_{g}"].mean() for g in range(g_max)])
#     sd_size_at = np.array([df[f"generation_{g}"].std() for g in range(g_max)])
#     extinct_se = np.sqrt(pr_extinct * (1.0 - pr_extinct)) / np.sqrt(n_sims_applicable)
#     size_se = sd_size_at / np.sqrt(n_sims_applicable)

#     return pl.DataFrame({
#         "generation" : gens,
#         "mean_size" : mean_size_at,
#         "prob_extinct" : pr_extinct,
#         "size_standard_error" : size_se,
#         "extinction_standard_error" : extinct_se,
#     })


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


def size_at_generation(sim: Simulation, generation: int) -> int:
    g_max = sim.params["n_generations"]
    assert (
        generation <= g_max
    ), "Generation size ill-defined for incompletely simulated generations"
    return len(sim.query_people({"generation": generation}))


def get_all_generation_counts(sims: Sequence[Simulation]) -> pl.DataFrame:
    all_gens = [sim.params["n_generations"] for sim in sims]
    assert (
        len(Counter(all_gens).items()) == 1
    ), "Cannot summarize simulations run for differing numbers of generations"
    g_max = all_gens[0] + 1
    return pl.concat(
        [
            pl.DataFrame(
                {"simulation": [idx]}
                | {
                    f"generation_{g}": [size_at_generation(sim, g)]
                    for g in range(g_max)
                }
            )
            for idx, sim in enumerate(sims)
        ]
    )


def summarize_generations(sims: Sequence[Simulation]) -> pl.DataFrame:
    df = get_all_generation_counts(sims)
    g_max = sims[0].params["n_generations"] + 1
    gens = np.arange(g_max)
    n_extinct_by = np.array([(df[f"generation_{g}"] == 0).sum() for g in range(g_max)])
    n_sims_applicable = df.shape[0] - n_extinct_by
    pr_extinct = n_extinct_by / len(sims)
    mean_size_at = np.array([df[f"generation_{g}"].mean() for g in range(g_max)])
    sd_size_at = np.array([df[f"generation_{g}"].std() for g in range(g_max)])
    extinct_se = np.sqrt(pr_extinct * (1.0 - pr_extinct)) / np.sqrt(n_sims_applicable)
    size_se = sd_size_at / np.sqrt(n_sims_applicable)

    return pl.DataFrame(
        {
            "generation": gens,
            "mean_size": mean_size_at,
            "prob_extinct": pr_extinct,
            "size_standard_error": size_se,
            "extinction_standard_error": extinct_se,
        }
    )


# def summarize_detections(sims: Sequence[Simulation]) -> pl.DataFrame:
#     per_sim = []
#     for sim in sims:
#         generation = []
#         detect_method, id = [], []
#         time_to_detection, time_to_infectious, duration_infectious = [], [], []
#         for infection in sim.infections:
#             print(f"+++ Type of generation is {type(infection['generation'])}")
#             generation.append(infection["generation"])
#             id.append(infection["id"])
#             time_to_infectious.append(infection["t_infectious"] - infection["t_exposed"])
#             if infection["detected"]:
#                 duration_infectious.append(infection["t_detected"] - infection["t_infectious"])
#                 time_to_detection.append(infection["t_detected"] - infection["t_exposed"])
#                 detect_method.append(infection["detect_method"])
#             else:
#                 duration_infectious.append(infection["t_recovered"] - infection["t_infectious"])
#                 time_to_detection.append(None)
#                 detect_method.append(None)

#         per_sim.append(pl.DataFrame({
#             "id" : id,
#             "generation" : generation,
#             "detect_method" : detect_method,
#             "time_to_detection" : time_to_detection,
#             "duration_infectious" : duration_infectious,
#             "time_to_infectious" : time_to_infectious,
#         }))

# return pl.concat(per_sim)

# def summarize_detections(sims: Sequence[Simulation]) -> pl.DataFrame:
#     cases, detect_active, detect_passive, before_infectious = 0, 0, 0, 0
#     time, time_active, time_passive, duration_infectious = 0.0, 0.0, 0.0, 0.0

#     for sim in sims:
#         for infection in sim.infections:
#             cases += 1
#             if infection["detected"]:
#                 time += infection["t_detected"] - infection["t_exposed"]
#                 duration_infectious += infection["t_detected"] - infection["t_infectious"]
#                 if infection["t_detected"] < infection["t_infectious"]:
#                     before_infectious += 1
#                 if infection["detect_method"] == "active":
#                     detect_active += 1
#                     time_active += infection["t_detected"] - infection["t_exposed"]
#                 elif infection["detect_method"] == "passive":
#                     detect_passive += 1
#                     time_passive += infection["t_detected"] - infection["t_exposed"]
#                 else:
#                     raise RuntimeError(f"Found unknown detection type {infection['detect_method']}")
#             else:
#                 duration_infectious

#     return pl.DataFrame({
#         "prob_active" : [detect_active / (cases - len(sims))], # index case can't be actively detected
#         "prob_passive" : [detect_passive / cases],
#     })
