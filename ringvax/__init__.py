from abc import ABC
from collections import namedtuple
from typing import Any, Iterable, Optional, Self, Sequence

import numpy as np

Event = namedtuple("Event", ["type", "time", "individual"])


class Individual(ABC):
    def __init__(
        self,
        params: dict[str, Any],
        time: float,
        ancestry: Sequence[Self],
        state: Optional[str] = None,
    ):
        self.state = state if state is not None else self.starting_state()
        self.params = params
        self.infected_time = time
        self.time = time
        self.ancestry: tuple[Self, ...] = tuple(ancestry)

    def apply_event(self) -> None:
        raise NotImplementedError()

    def next_event(self, **kwargs) -> Event:
        raise NotImplementedError()

    def starting_state(self) -> str:
        raise NotImplementedError()

    def validate_params(self) -> None:
        pass


class BdsSeirIndividual(Individual):
    par_names = ("birth_rate", "death_rate", "sampling_rate", "ei_rate")
    par_types = (float,) * len(par_names)

    def validate_params(self):
        for p, t in zip(
            BdsSeirIndividual.par_names, BdsSeirIndividual.par_types
        ):
            assert p in self.params and isinstance(self.params[p], t)


class InfectionHandler(ABC):
    def __call__(self, infector: Individual, **kwargs) -> Individual:
        raise NotImplementedError()


class Population(ABC):
    def __init__(
        self, init_infecteds: Iterable[Individual], time: float = 0.0
    ):
        self.infecteds = list(init_infecteds)
        self.deterministic_event_stack: list[Event] = list()
        self.time = time

    # Recover, vaccinate, etc.
    def handle_next_event(self, individual: Individual, event: Event) -> None:
        raise NotImplementedError()

    def next_event(self) -> tuple[Individual, Event]:
        raise NotImplementedError()

    def next_infection_to(self, infector: Individual) -> Individual:
        raise NotImplementedError()

    def num_infected(self) -> int:
        raise NotImplementedError()

    def ring_vaccinate(
        self, order, target: Individual, ve: float, rampup: float, **kwargs
    ) -> None:
        for individ in self.get_ring(order, target, **kwargs):
            self.vaccinate(individ, ve, rampup)

    def vaccinate(
        self, individual: Individual, ve: float, rampup: float
    ) -> None:
        # "all or nothing vaccination"
        if np.random.uniform(0, 1) <= ve:
            self.deterministic_event_stack.append(
                Event(
                    type="vaccination",
                    time=self.time + rampup,
                    individual=individual,
                )
            )

    def get_ring(
        self, order: int, individual: Individual, **kwargs
    ) -> Iterable[Individual]:
        # Depends on:
        #    - population model, e.g. in a branching process, this is just the already-infected children
        #    - detection model, e.g., do we see everyone or not?
        raise NotImplementedError()
