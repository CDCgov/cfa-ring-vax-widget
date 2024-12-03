import numpy as np
import streamlit as st

import ringvax


def app():
    st.title("Ring vaccination efficacy simulator")
    st.write(
        """We assume that all detected infections are instantaneously quarantined with perfect efficacy."""
    )

    # Sidebar for inputs
    st.sidebar.header("Natural history parameters")
    r_0 = st.sidebar.number_input("R0", value=1.0)
    exposed_duration = st.sidebar.number_input(
        "Pre-infectious duration (in days)",
        value=1,
        min_value=0,
        max_value=365,
        step=1,
    )
    infectious_duration = st.sidebar.number_input(
        "Infectious duration (in days)",
        value=1,
        min_value=0,
        max_value=365,
        step=1,
    )
    passive_detect_prob = st.sidebar.number_input(
        "Probability of passive detection",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    contact_detect_prob = st.sidebar.number_input(
        "Probability a contact is found",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    g_max = st.sidebar.number_input(
        "Simulation duration (in expected generations)",
        value=4.0,
        min_value=0.0,
        max_value=100.0,
        step=1.0,
    )

    death_rate = 1.0 / infectious_duration
    birth_rate = r_0 * death_rate
    sampling_rate = passive_detect_prob * death_rate
    ei_rate = 1.0 / exposed_duration

    st.write(
        f"Parameters: lambda = {birth_rate}; mu = {death_rate}; psi = {sampling_rate}; q_ei = {ei_rate}; contact_trace_prob = {contact_detect_prob}"
    )
    nsim = 1000

    # Simulations

    # Outbreak sizes in the counterfactual
    no_intervention_sizes = []
    for idx in range(nsim):
        no_intervention_sizes.append(
            len(
                ringvax.BirthDeathSimulator()(
                    ringvax.BirthDeathParticlePool(
                        birth_rate,
                        death_rate,
                        0.0,
                        ei_rate,
                        rng=np.random.default_rng(idx),
                    ),
                    ring_vax_detect_prob=0.0,
                    t_max=g_max * infectious_duration,
                ).pool.pool
            )
        )
    no_intervention_sizes = np.array(no_intervention_sizes)

    # Outbreak sizes with detection and contact tracing
    sizes = []
    for _ in range(nsim):
        sizes.append(
            len(
                ringvax.BirthDeathSimulator()(
                    ringvax.BirthDeathParticlePool(
                        birth_rate,
                        death_rate,
                        sampling_rate,
                        ei_rate,
                        rng=np.random.default_rng(idx),
                    ),
                    ring_vax_detect_prob=contact_detect_prob,
                    t_max=g_max * infectious_duration,
                ).pool.pool
            )
        )
    sizes = np.array(sizes)

    if passive_detect_prob > 0.0:
        # How many generations past the index case are we (when we find the index case)?
        first_detection_generations = []
        sim_idx = 0
        while len(first_detection_generations) < nsim:
            sim = ringvax.BirthDeathSimulator()(
                ringvax.BirthDeathParticlePool(
                    birth_rate,
                    death_rate,
                    sampling_rate,
                    ei_rate,
                    rng=np.random.default_rng(sim_idx),
                ),
                ring_vax_detect_prob=contact_detect_prob,
                t_max=g_max * infectious_duration,
                stop_condition="detect_index_passive",
            )
            sim_idx += 1
            if sim.condition_success:
                first_detection_generations.append(
                    ringvax.count_generations(
                        0, [*sim.pool.pool, *sim.pool.removed]
                    )
                )
        first_detection_generations = np.array(first_detection_generations)

    # Display results
    st.header("Results")

    st.subheader(
        f"Proportion extinct after time {g_max * infectious_duration}"
    )
    st.write(
        f"Without detection or contact tracing: {(no_intervention_sizes == 0.0).mean():.2f}"
    )
    st.write(
        f"With detection and contact tracing: {(sizes == 0.0).mean():.2f}"
    )

    st.subheader(
        f"Average number of infections after time {g_max * infectious_duration}"
    )
    st.write(
        f"Without detection or contact tracing: {no_intervention_sizes.mean():.2f}"
    )
    st.write(f"With detection and contact tracing: {sizes.mean():.2f}")

    if passive_detect_prob > 0.0:
        st.subheader(
            "Maximum number of generations when index case is detected"
        )
        st.write(f"Mean: {first_detection_generations.mean():.2f}")
        st.write(
            f"Median: {np.quantile(first_detection_generations, 0.5):.2f}"
        )
        st.write(f"90%: {np.quantile(first_detection_generations, 0.9):.2f}")


if __name__ == "__main__":
    app()
