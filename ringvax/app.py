import graphviz
import streamlit as st

from ringvax import Simulation


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
    max_infections = st.number_input(
        "Maximum number of infections", value=100, step=10, min_value=10
    )
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
        "max_infections": max_infections,
    }

    st.subheader(
        f"R0 is {infectious_duration * infection_rate:.2f}",
        help="R0 is the average duration of infection multiplied by the infectious rate.",
    )

    s = Simulation(params=params, seed=seed)
    s.run()

    st.header("Graph of infections")
    st.graphviz_chart(make_graph(s))

    st.header("Raw results")
    for id, content in s.infections.items():
        st.text(id)
        st.text(content)


if __name__ == "__main__":
    app()
