import altair as alt
import graphviz
import polars as pl
import streamlit as st

from ringvax import Simulation
from ringvax.summary import (
    get_all_person_properties,
    summarize_detections,
    summarize_generations,
    summarize_infections,
)


def make_graph(sim: Simulation):
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


def app():
    st.title("Ring vaccination")

    with st.sidebar:
        latent_duration = st.slider(
            "Latent duration",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f days",
        )
        infectious_duration = st.slider(
            "Infectious duration",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            format="%.1f days",
        )
        infection_rate = st.slider(
            "Infection rate", min_value=0.0, max_value=10.0, value=0.5, step=0.1
        )
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
        n_generations = st.number_input("Number of generations", value=4, step=1)
        max_infections = st.number_input(
            "Maximum number of infections", value=100, step=10, min_value=10
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

    sims = []
    for i in range(nsim):
        sims.append(Simulation(params=params, seed=seed + i))
        sims[-1].run()

    st.text(f"Ran {nsim} simulations")

    st.subheader(
        f"R0 is {infectious_duration * infection_rate:.2f}",
        help="R0 is the average duration of infection multiplied by the infectious rate.",
    )

    tab1, tab2 = st.tabs(["Simulation summary", "Per-simulation results"])
    with tab1:
        sim_df = get_all_person_properties(sims)

        st.header("Summary of dynamics")
        infection = summarize_infections(sim_df)
        st.write(
            f"In these simulations, the average duration of infectiousness was {infection['mean_infectious_duration'][0]:.2f} and $R_e$ was {infection['mean_n_infections'][0]:.2f}"
        )

        st.write(
            "The following table provides summaries of _marginal_ probabilities regarding detection."
        )
        detection = summarize_detections(sim_df)
        st.dataframe(
            detection.rename(
                {
                    "prob_detect": "Any detection",
                    "prob_active": "Active detection",
                    "prob_passive": "Passive detection",
                    "prob_detect_before_infectious": "Detection before onset of infectiousness",
                }
            )
        )

        st.header("Extinction probability by generation")
        df = summarize_generations(sims).with_columns(
            mean_size_low=(pl.col("mean_size") - pl.col("size_standard_error")),
            mean_size_high=(pl.col("mean_size") + pl.col("size_standard_error")),
            prob_extinct_low=(
                pl.col("prob_extinct") - pl.col("extinction_standard_error")
            ),
            prob_extinct_high=(
                pl.col("prob_extinct") + pl.col("extinction_standard_error")
            ),
        )

        size_plot = alt.Chart(df)

        # Plot containment probability
        points = size_plot.mark_point(filled=True, size=50, color="black").encode(
            alt.X("generation"), alt.Y("prob_extinct")
        )

        # generate the error bars
        errorbars = size_plot.mark_errorbar().encode(
            x=alt.X("generation", title="Generation"),
            y=alt.Y("prob_extinct_low:Q", title="Cumulative containment probability"),
            y2="prob_extinct_high:Q",
        )

        st.altair_chart((points + errorbars))  # type: ignore

        st.header("Number of infections by generation")
        # Plot average size
        points = size_plot.mark_point(filled=True, size=50, color="black").encode(
            alt.X("generation"), alt.Y("mean_size")
        )

        # generate the error bars
        errorbars = size_plot.mark_errorbar().encode(
            x=alt.X("generation", title="Generation"),
            y=alt.Y("mean_size_low:Q", title="Mean number of infections"),
            y2="mean_size_high:Q",
        )

        st.altair_chart((points + errorbars))  # type: ignore

    with tab2:
        st.header("Graph of infections")
        idx = st.number_input("Simulation to plot", min_value=0, max_value=nsim, value=0)
        st.graphviz_chart(make_graph(sims[idx]))


if __name__ == "__main__":
    app()
