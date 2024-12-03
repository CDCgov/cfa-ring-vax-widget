from collections import namedtuple
from copy import deepcopy
from typing import Iterable, Sequence

import numpy as np

# Particles, while simple, should be mutable to avoid painful unneeded copying and to preserve infection trees
Particle = namedtuple("Particle", ["id", "state", "infector", "infectees"])

SimulatorResults = namedtuple(
    "SimulatorResults", ["pool", "condition_success"]
)


def count_generations(id, pool: Iterable[Particle]) -> int:
    particle = next((part for part in pool if part.id == id), None)
    assert (
        particle is not None
    ), f"Cannot find particle with id {id} in pool {pool}"
    if len(particle.infectees) == 0:
        return 0
    else:
        tip_gens = []
        _count_generations(particle, pool, 0, tip_gens)
        return max(tip_gens)


def _count_generations(
    particle: Particle, pool: Iterable[Particle], gen, gens: list
) -> None:
    if len(particle.infectees) == 0:
        gens.append(gen)
    else:
        for infectee in particle.infectees:
            id = infectee.id
            child = next((part for part in pool if part.id == id), None)
            assert child is not None
            _count_generations(child, pool, gen + 1, gens)


class BirthDeathParticlePool:
    states = ["exposed", "infectious"]
    events = ["state_change", "birth", "death", "detection"]

    def __init__(
        self,
        per_capita_birth_rate: float,
        per_capita_death_rate: float,
        per_capita_sampling_rate: float,
        per_capita_ei_rate: float,
        recover_on_sample_prob: float = 1.0,
        infected_ids: Sequence[int] = [0],
        states: Sequence[str] = ["exposed"],
        rng=np.random.default_rng(),
    ):
        assert len(states) == len(
            infected_ids
        ), "Must provide exactly one state per infected"
        assert all(state in BirthDeathParticlePool.states for state in states)

        self.birth_rate = per_capita_birth_rate
        self.death_rate = per_capita_death_rate
        self.sampling_rate = per_capita_sampling_rate
        self.ei_rate = per_capita_ei_rate
        self.recover_on_sample_prob = recover_on_sample_prob
        self.rng = rng
        self.pool: tuple[Particle, ...] = tuple(
            Particle(id=id, state=state, infector=None, infectees=tuple())
            for id, state in zip(infected_ids, states)
        )
        self.removed: tuple[Particle, ...] = tuple()

    def __contains__(self, particle: Particle):
        for part in self.pool:
            if part.id == particle.id:
                return True
        return False

    def birth(self, parent=None) -> tuple[str, Particle]:
        if parent is None:
            parent = self.rand_particle("infectious")
        child = Particle(
            id=self.next_id(),
            state="exposed",
            infector=parent,
            infectees=tuple(),
        )
        new_parent = Particle(
            id=parent.id,
            state=parent.state,
            infector=parent.infector,
            infectees=parent.infectees + (child,),
        )

        self.pool = tuple(p for p in self.pool if p is not parent) + (
            new_parent,
            child,
        )

        return (
            "birth",
            child,
        )

    def death(self, particle=None) -> tuple[str, Particle]:
        if particle is None:
            particle = self.rand_particle("infectious")

        self.pool = tuple(p for p in self.pool if p is not particle)
        self.removed = self.removed + (particle,)
        return (
            "death",
            particle,
        )

    def detection(self, particle=None) -> tuple[str, Particle]:
        if particle is None:
            particle = self.rand_particle("infectious")
        if (
            self.rng.uniform(low=0.0, high=1.0, size=1)
            < self.recover_on_sample_prob
        ):
            return ("detection", self.death(particle)[1])
        else:
            return ("detection", particle)

    def draw_waiting_time(self):
        assert len(self.pool) > 0
        n_e = sum(1 for part in self.pool if part.state == "exposed")
        n_i = len(self.pool) - n_e
        total_rate = (
            n_i * (self.birth_rate + self.death_rate + self.sampling_rate)
            + n_e * self.ei_rate
        )
        return self.rng.exponential(scale=1.0 / total_rate, size=1)

    def next_id(self):
        return max(part.id for part in self.pool) + 1

    def rand_particle(self, state):
        assert len(self.pool) > 0
        ids = [part.id for part in self.pool if part.state == state]
        id = self.rng.choice(ids)
        particle = next((part for part in self.pool if part.id == id), None)
        assert isinstance(particle, Particle)
        return particle

    def resolve_event(self):
        assert len(self.pool) > 0
        n_e = sum(1 for part in self.pool if part.state == "exposed")
        n_i = len(self.pool) - n_e
        rates = np.array(
            [
                n_e * self.ei_rate,
                n_i * self.birth_rate,
                n_i * self.death_rate,
                n_i * self.sampling_rate,
            ]
        )
        event = self.rng.choice(
            BirthDeathParticlePool.events,
            p=rates / rates.sum(),
        )
        return getattr(self, event)()

    def state_change(self, exposed=None):
        if exposed is None:
            exposed = self.rand_particle("exposed")

        infectious = Particle(
            id=exposed.id,
            state="infectious",
            infector=exposed.infector,
            infectees=exposed.infectees,
        )
        self.pool = tuple(p for p in self.pool if p is not exposed) + (
            infectious,
        )
        return (
            "state_change",
            infectious,
        )


def ring_vaccinate(particle, pool, ring_vax_detect_prob):
    contacts = list(particle.infectees)
    if particle.infector is not None:
        contacts.append(particle.infector)

    while len(contacts) > 0:
        contact = contacts.pop()
        # particle's parent is a contact, but particle has already been handled
        if (contact is not particle) and (
            pool.rng.uniform(0.0, 1.0, size=1) < ring_vax_detect_prob
        ):
            # Use .detection() rather than .death() because .detection() includes possibility of nonremoval
            contacts = [*contacts, *contact.infectees]
        # Can't remove an already-removed particle
        if contact in pool:
            _ = pool.detection(contact)


class BirthDeathSimulator:
    def __call__(
        self,
        starting_pool: BirthDeathParticlePool,
        ring_vax_detect_prob: float,
        t_max: float,
        stop_condition="time",
    ) -> SimulatorResults:
        pool = deepcopy(starting_pool)
        time = 0.0
        condition_success = False

        while True:
            wt = pool.draw_waiting_time()
            if time + wt >= t_max:
                break

            time = time + wt
            event, particle = pool.resolve_event()

            if event == "detection" and ring_vax_detect_prob > 0.0:
                ring_vaccinate(particle, pool, ring_vax_detect_prob)
                if (
                    particle.id == 0
                    and stop_condition == "detect_index_passive"
                ):
                    condition_success = True
                    break

            if len(pool.pool) == 0:
                break

        if stop_condition == "time":
            condition_success = True if np.isclose(time, t_max) else False

        return SimulatorResults(pool=pool, condition_success=condition_success)
