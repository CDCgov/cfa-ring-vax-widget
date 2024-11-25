from abc import ABC
from collections import namedtuple
from typing import Any, Iterable, Optional, Self, Sequence

import numpy as np

Event = namedtuple("Event", ["type", "time", "individual"])
"""
Events are things like "individual X causes an infection at time T"
"""


class Individual(ABC):
    """
    An infected individual which can infect others.
    """

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

    def apply_event(self, Event) -> None:
        """
        Resolve event, e.g. moving from Exposed to Infectious
        """
        raise NotImplementedError()

    def generate_infection(self, time: float) -> Self:
        """
        Get a new infected resulting from this infector at given time.
        """
        raise NotImplementedError()

    def next_event(self, **kwargs) -> Event:
        """
        What is the next Event for this individual.
        """
        raise NotImplementedError()

    def starting_state(self) -> str:
        """
        Default starting state for an individual, e.g. I for SIR, E for SEIR
        """
        raise NotImplementedError()

    def validate_params(self) -> None:
        pass


class Population(ABC):
    def __init__(
        self, init_infecteds: Iterable[Individual], time: float = 0.0
    ):
        self.infecteds = list(init_infecteds)
        self.event_stack: list[Event] = list()
        self.time = time

    # Recover, vaccinate, etc.
    def handle_next_event(self, individual: Individual, event: Event) -> None:
        raise NotImplementedError()

    def next_individual_level_event(self) -> tuple[Individual, Event]:
        """
        Find next event (presumably but not necessarily leaning on self.infecteds' next_event())
        """
        raise NotImplementedError()

    def next_infection_to(self, infector: Individual) -> Individual:
        """
        What individual is infected by stated infector?
        """
        raise NotImplementedError()

    def ring_vaccinate(
        self,
        max_degree,
        target: Individual,
        ve: float,
        rampup: float,
        **kwargs
    ) -> None:
        """
        Get ring around individual, apply vaccines
        """
        for individ in self.get_ring(max_degree, target, **kwargs):
            self.vaccinate(individ, ve, rampup)

    def step(self):
        """
        Advance simulation by one event
        """

    def vaccinate(self, target: Individual, ve: float, rampup: float) -> None:
        """
        Register with population intent to apply "all or nothing" vaccination to target individual.

        Individual will be removed with probability given by VE when vaccine becomes effective (given by rampup).
        """
        # "all or nothing vaccination"
        if np.random.uniform(0, 1) <= ve:
            self.event_stack.append(
                Event(
                    type="vaccination",
                    time=self.time + rampup,
                    individual=target,
                )
            )

    def get_ring(
        self, max_degree: int, target: Individual, **kwargs
    ) -> Iterable[Individual]:
        """
        Get contacts (and potentially contacts of contacts of... out to max_degree) target individual.
        """
        # Depends on:
        #    - population model, e.g. in a branching process, this is just the already-infected children
        #    - detection model, e.g., do we see everyone or not?
        raise NotImplementedError()
