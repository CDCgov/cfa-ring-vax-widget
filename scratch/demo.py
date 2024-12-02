import numpy as np

import ringvax

nrep = 100

pool = ringvax.BirthDeathParticlePool(2, 1, 1, 1)

no_intervention_sizes = []
for _ in range(nrep):
    no_intervention_sizes.append(
        len(ringvax.BirthDeathSimulator()(pool, 0.0, 4.0).pool)
    )
no_intervention_sizes = np.array(no_intervention_sizes)

no_intervention_sizes.mean()
(no_intervention_sizes == 0).mean()


with_intervention_sizes = []
for _ in range(nrep):
    with_intervention_sizes.append(
        len(ringvax.BirthDeathSimulator()(pool, 0.25, 4.0).pool)
    )
with_intervention_sizes = np.array(with_intervention_sizes)
