import altair as alt
import polars as pl
import streamlit as st

from ringvax import Simulation
from ringvax.summary import summarize_generations


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
            "Infection rate", min_value=0.0, max_value=10.0, value=1.0, step=0.1
        )
        p_passive_detect = (
            st.slider(
                "Passive detection probability",
                min_value=0.0,
                max_value=100.0,
                value=0.5,
                step=0.01,
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
                step=0.1,
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
        seed = st.number_input("Random seed for first simulation", value=42, step=1)
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
        st.write("Under construction!")


if __name__ == "__main__":
    app()
