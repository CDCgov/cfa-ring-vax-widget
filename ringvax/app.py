import time
from typing import List

import altair as alt
import graphviz
import numpy as np
import numpy.random
import polars as pl
import streamlit as st

from ringvax import Simulation
from ringvax.summary import (
    get_all_person_properties,
    get_total_infection_count_df,
    prob_control_by_gen,
    summarize_detections,
    summarize_infections,
)


def render_percents(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(col).is_nan())
        .then(pl.lit("Not a number"))
        .otherwise(
            pl.col(col).map_elements(
                lambda x: f"{round(x):.0f}%", return_dtype=pl.String
            )
        )
        .alias(col)
        for col in df.columns
    )


def make_graph(sim: Simulation) -> graphviz.Digraph:
    """Make a transmission graph"""
    graph = graphviz.Digraph()
    for infectee in sim.query_people():
        infector = sim.get_person_property(infectee, "infector")

        color = (
            "black" if not sim.get_person_property(infectee, "detected") else "#068482"
        )

        graph.node(str(infectee), color=color)

        if infector is not None:
            graph.edge(str(infector), str(infectee))

    return graph


@st.fragment
def show_graph(sims: List[Simulation], pause: float = 0.1):
    """Show a transmission graph. Wrap as st.fragment, to not re-run simulations.

    Args:
        sims (List[Simulation]): list of simulations
        pause (float, optional): Number of seconds to pause before displaying
            new graph. Defaults to 0.1.
    """
    idx = st.number_input(
        "Simulation to plot", min_value=0, max_value=len(sims) - 1, value=0
    )
    placeholder = st.empty()
    time.sleep(pause)
    placeholder.graphviz_chart(make_graph(sims[idx]))


def format_control_gens(gen: int):
    if gen == 0:
        return "index_case"
    if gen == 1:
        return "contacts"
    elif gen > 1:
        return "".join(["contacts of "] * (gen - 1)) + "contacts"
    else:
        raise RuntimeError("Must specify `gen` >= 0.")


def format_duration(x: float, digits=3) -> str:
    """Format a number of seconds duration into a string"""
    assert x >= 0
    min_time = 10 ** (-digits)
    if x < min_time:
        return f"<{min_time} seconds"
    else:
        return f"{round(x, digits)} seconds"


def set_session_default(key, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def app():
    st.info(
        "This interactive application is a prototype designed for software testing and educational purposes."
    )

    st.title("Ring vaccination")

    with st.sidebar:
        st.subheader("Disease history times")
        latent_duration = st.slider(
            "Latent duration",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f days",
        )

        default_infectious_duration = 3.0
        max_infectious_duration = 10.0
        min_infectious_duration = 0.1

        default_R0 = 1.5
        max_R0 = 10.0
        set_session_default("R0", default_R0)
        set_session_default("infection_rate", default_R0 / default_infectious_duration)

        def infectiousness_callback():
            """
            Update either R0 or rate slider, based on the infectious duration, the other
            value, and which one of the values is taken as fixed.
            """
            control = st.session_state["infectiousness_control"]
            if control == "R0":
                st.session_state["infection_rate"] = (
                    st.session_state["R0"] / st.session_state["infectious_duration"]
                )
            elif control == "rate":
                st.session_state["R0"] = (
                    st.session_state["infection_rate"]
                    * st.session_state["infectious_duration"]
                )
            else:
                raise RuntimeError(f"Unknown {control=}")

        infectious_duration = st.slider(
            "Infectious duration",
            key="infectious_duration",
            min_value=min_infectious_duration,
            max_value=max_infectious_duration,
            value=default_infectious_duration,
            step=0.1,
            format="%.1f days",
            on_change=infectiousness_callback,
        )

        st.subheader("Infectiousness")

        infectiousness_control = st.segmented_control(
            "Variable infectiousness parameter",
            key="infectiousness_control",
            options=["R0", "rate"],
            selection_mode="single",
            default="R0",
            help="R0 = infectious duration * infection rate. Only two of the "
            "three parameters can be varied.",
        )

        R0 = st.slider(
            "R0",
            key="R0",
            min_value=0.0,
            max_value=max_R0,
            step=0.1,
            disabled=infectiousness_control != "R0",
            on_change=infectiousness_callback,
        )

        infection_rate = st.slider(
            "Infection rate (mean infections per day)",
            key="infection_rate",
            min_value=0.0,
            max_value=max_R0 / min_infectious_duration,
            step=0.1,
            disabled=infectiousness_control != "rate",
            on_change=infectiousness_callback,
        )

        # double check that R0, rate, and duration are not in conflict
        assert np.isclose(R0, infection_rate * infectious_duration)

        # rather than trying to dynamically adjust the ranges on the sliders,
        # issue a warning if the selected rate leads to an out-of-range R0
        if R0 > max_R0:
            st.warning(f"Selected infectious rate yields R0 > {max_R0}")

        st.subheader("Detection")
        p_passive_detect = (
            st.slider(
                "Passive detection probability",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                format="%d%%",
            )
            / 100.0
        )
        passive_detection_delay = st.slider(
            "Passive detection delay",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f days",
        )
        p_active_detect = (
            st.slider(
                "Active detection probability",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                format="%d%%",
            )
            / 100.0
        )
        active_detection_delay = st.slider(
            "Active detection delay",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f days",
        )

        with st.expander("Advanced Options"):
            n_generations = st.number_input(
                "Number of simulated generations", value=4, step=1
            )
            control_generations = st.number_input(
                "Degree of contacts for checking control",
                value=3,
                step=1,
                min_value=1,
                max_value=n_generations + 1,
                help="Successful control is defined as no infections in contacts at this degree. Set to 1 for contacts of the index case, 2 for contacts of contacts, etc. Equivalent to checking for extinction in the specified generation.",
            )
            max_infections = st.number_input(
                "Maximum number of infections",
                value=1000,
                step=10,
                min_value=100,
                help="",
            )
            seed = st.number_input("Random seed", value=1234, step=1)
            nsim = st.number_input("Number of simulations", value=250, step=1)

    params = {
        "n_generations": n_generations,
        "latent_duration": latent_duration,
        "infectious_duration": infectious_duration,
        "infection_rate": infection_rate,
        "p_passive_detect": p_passive_detect,
        "passive_detection_delay": passive_detection_delay,
        "p_active_detect": p_active_detect,
        "active_detection_delay": active_detection_delay,
        "max_infections": max_infections,
    }

    progress_text = (
        "Running simulation... Slow simulations may indicate unreasonable "
        "parameter values leading to unrealistically large total numbers of "
        "infections."
    )
    progress_bar = st.progress(0, text=progress_text)

    # run simulations ---------------------------------------------------------
    tic = time.perf_counter()
    sims = []

    # initialize rngs
    rngs = numpy.random.default_rng(seed).spawn(nsim)

    for i in range(nsim):
        progress_bar.progress(i / nsim, text=progress_text)
        sim = Simulation(params=params, rng=rngs[i])
        sim.run()
        sims.append(sim)

    progress_bar.empty()
    toc = time.perf_counter()
    # end simulations ---------------------------------------------------------

    st.write(f"Ran {nsim} simulations in {format_duration(toc - tic)}")

    n_at_max = sum(1 for sim in sims if sim.termination == "max_infections")

    show = True if n_at_max == 0 else False
    if not show:
        st.warning(
            body=(
                f"{n_at_max} simulations hit the specified maximum number of infections ({max_infections})."
            ),
            icon="🚨",
        )

        st.warning(
            body=(
                "Simulations hitting the maximum likely indicate implausible parameter values. "
                'It is recommended that you either adjust simulating parameters or increase "Maximum number of infections".'
            ),
        )

        st.warning(
            body=(
                "Note that results are summarized only for simulations which do not exceed this maximum. "
                "This means that simulations with large final sizes will be missing from the results, biasing results. "
            ),
        )

        accept_terms_and_conditions = st.button(
            "I accept that the results are biased and may not be meaningful. Please show them anyways."
        )
        if accept_terms_and_conditions:
            show = True

    if show:
        if n_at_max == nsim:
            st.error(
                "No simulations completed successfully. Please change settings and try again.",
                icon="🚨",
            )
            st.stop()

        tab1, tab2 = st.tabs(["Simulation summary", "Per-simulation results"])
        with tab1:
            sim_df = get_all_person_properties(sims)

            pr_control = prob_control_by_gen(sim_df, control_generations)
            st.header(
                f"Probability of control: {pr_control:.0%}",
                help=f"The probability that there are no infections in the {format_control_gens(control_generations)}, or equivalently that the {format_control_gens(control_generations - 1)} do not produce any further infections.",
            )

            st.header("Number of infections")
            st.write(
                f"Distribution of the total number of infections seen in {n_generations} generations."
            )
            st.altair_chart(
                alt.Chart(get_total_infection_count_df(sim_df))
                .mark_bar()
                .encode(
                    x=alt.X("size:Q", bin=True, title="Number of infections"),
                    y=alt.Y("count()", title="Count"),
                )
            )

            st.header("Summary of dynamics")
            infection = summarize_infections(sim_df)
            st.write(
                f"In these simulations, the average duration of infectiousness was {infection['mean_infectious_duration'][0]:.2f} and $R_e$ was {infection['mean_n_infections'][0]:.2f}"
            )

            st.write(
                (
                    "The following table provides summaries of marginal probabilities regarding detection. "
                    "Aside from the marginal probability of active detection, these are the observed "
                    "probabilities that any individual is detected in this manner, including the index case. "
                    "The marginal probability of active detection excludes index cases, which are not "
                    "eligible for active detection."
                )
            )
            detection = summarize_detections(sim_df)
            st.dataframe(
                render_percents(detection).rename(
                    {
                        "prob_detect": "Any detection",
                        "prob_active": "Active detection",
                        "prob_passive": "Passive detection",
                        "prob_detect_before_infectious": "Detection before onset of infectiousness",
                    }
                )
            )

        with tab2:
            st.header("Graph of infections")
            show_graph(sims=sims)


if __name__ == "__main__":
    app()
