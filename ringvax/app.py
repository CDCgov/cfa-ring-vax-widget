import streamlit as st

from ringvax import Simulation


def app():
    st.title("Ring vaccination")

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
    seed = st.number_input("Random seed", value=1234, step=1)

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
    s = Simulation(params=params, seed=seed)
    s.run()
    st.text(s.infections)


if __name__ == "__main__":
    app()
