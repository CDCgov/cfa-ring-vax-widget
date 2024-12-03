import numpy as np

import ringvax

nrep = 1000

r_0 = 2.0
infectious_duration = 5.0
exposed_duration = 2.0
sampling_prob = 0.5

death_rate = 1.0 / infectious_duration
birth_rate = r_0 * death_rate
sampling_rate = sampling_prob * death_rate
ei_rate = 1.0 / exposed_duration

print(
    f">> With only passive detection, Re = {birth_rate / (death_rate + sampling_rate)} <<"
)

first_detection_generations = []
sim_idx = 0
while len(first_detection_generations) < nrep:
    sim = ringvax.BirthDeathSimulator()(
        ringvax.BirthDeathParticlePool(
            birth_rate,
            death_rate,
            sampling_rate,
            ei_rate,
            rng=np.random.default_rng(sim_idx),
        ),
        ring_vax_detect_prob=0.25,
        t_max=4 * infectious_duration,
        stop_condition="detect_index_passive",
    )
    sim_idx += 1
    if sim.condition_success:
        first_detection_generations.append(
            ringvax.count_generations(0, [*sim.pool.pool, *sim.pool.removed])
        )
first_detection_generations = np.array(first_detection_generations)

first_detection_generations.mean()
(first_detection_generations == 0).mean()
(first_detection_generations >= 2).mean()

no_intervention_sizes = []
for idx in range(nrep):
    no_intervention_sizes.append(
        len(
            ringvax.BirthDeathSimulator()(
                ringvax.BirthDeathParticlePool(
                    birth_rate,
                    death_rate,
                    sampling_rate,
                    ei_rate,
                    rng=np.random.default_rng(idx),
                ),
                ring_vax_detect_prob=0.0,
                t_max=2 * infectious_duration,
            ).pool.pool
        )
    )
no_intervention_sizes = np.array(no_intervention_sizes)

with_intervention_sizes = []
for _ in range(nrep):
    with_intervention_sizes.append(
        len(
            ringvax.BirthDeathSimulator()(
                ringvax.BirthDeathParticlePool(
                    birth_rate, death_rate, sampling_rate, ei_rate
                ),
                ring_vax_detect_prob=0.25,
                t_max=2 * infectious_duration,
            ).pool.pool
        )
    )
with_intervention_sizes = np.array(with_intervention_sizes)

no_intervention_sizes.mean()
(no_intervention_sizes == 0).mean()


with_intervention_sizes.mean()
(with_intervention_sizes == 0).mean()
