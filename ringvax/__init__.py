from collections import namedtuple
from copy import copy
from typing import Sequence

import numpy as np

Particle = namedtuple("Particle", ["id", "state", "infector", "infectees"])


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
        self.pool: list[Particle] = [
            Particle(id=id, state=state, infector=None, infectees=[])
            for id, state in zip(infected_ids, states)
        ]

    def birth(self, parent=None) -> tuple[str, Particle]:
        if parent is None:
            parent = self.rand_particle("infectious")
        child = Particle(
            id=self.next_id(), state="exposed", infector=parent, infectees=[]
        )
        self.pool.append(child)
        return (
            "birth",
            child,
        )

    def death(self, particle=None) -> tuple[str, Particle]:
        if particle is None:
            particle = self.rand_particle("infectious")
        return (
            "death",
            self.pool.pop(self.pool.index(particle)),
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

    def state_change(self, particle=None):
        if particle is None:
            particle = self.rand_particle("exposed")
        exposed = self.pool.pop(self.pool.index(particle))
        infectious = Particle(
            id=exposed.id,
            state="infectious",
            infector=exposed.infector,
            infectees=exposed.infectees,
        )
        self.pool.append(infectious)
        return (
            "state_change",
            infectious,
        )


class BirthDeathSimulator:
    def __call__(
        self,
        starting_pool: BirthDeathParticlePool,
        ring_vax_detect_prob: float,
        t_stop: float,
    ):
        pool = copy(starting_pool)
        pool.pool = copy(starting_pool.pool)
        rng = pool.rng
        time = 0.0

        while True:
            wt = pool.draw_waiting_time()
            if time + wt >= t_stop:
                break

            time = time + wt
            event, particle = pool.resolve_event()

            if event == "detection" and ring_vax_detect_prob > 0.0:
                contacts = [particle.infector, *particle.infectees]
                while len(contacts) > 0:
                    contact = contacts.pop()
                    if rng.uniform(0.0, 1.0, size=1) < ring_vax_detect_prob:
                        # Use .detection() rather than .death() because .detection() includes possibility of nonremoval
                        _ = pool.detection(contact)
                        contacts = [*contacts, *contact.infectees]

            if len(pool.pool) == 0:
                break

        return pool
