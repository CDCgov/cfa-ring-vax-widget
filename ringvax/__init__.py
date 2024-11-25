import math
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Iterable, Optional, Self, Sequence

import numpy as np

Event = namedtuple("Event", ["type", "time", "individual"])
"""
Events are things like "individual X causes an infection at time T"

Conventions for types:
  infection: a new infection arises
  recovery: an individual becomes non-infectious
  detection: a pre-existing infection is detected
  transition_[from]_[to]: moving between states of infection, such as exposure to infectiousness
"""


class EventArgMin(ABC):
    """
    From a set of events, determine which happens first.

    This handles time tie-breaking via event prioritization and as-needed randomization
    """

    @abstractmethod
    def __call__(self, events: Iterable[Event]) -> Event:
        raise NotImplementedError()


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

    @abstractmethod
    def apply_event(self, Event) -> None:
        """
        Resolve event, e.g. moving from Exposed to Infectious
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_infection(self, time: float) -> Self:
        """
        Get a new infected resulting from this infector at given time.
        """
        raise NotImplementedError()

    @abstractmethod
    def next_event(self, **kwargs) -> Event:
        """
        What is the next Event for this individual.
        """
        raise NotImplementedError()

    @abstractmethod
    def starting_state(self) -> str:
        """
        Default starting state for an individual, e.g. I for SIR, E for SEIR
        """
        raise NotImplementedError()

    def validate_params(self) -> None:
        """
        Check arguments are valid, error if not, default is permissive
        """
        pass


class Population(ABC):
    """
    A group of Individuals, plus emergent properties.
    """

    def __init__(
        self,
        init_infecteds: Iterable[Individual],
        next_event: EventArgMin,
        time: float = 0.0,
    ):
        self.infecteds = list(init_infecteds)
        self.event_stack: list[Event] = list(
            Event(type="dummy", time=math.inf, individual=None)
        )
        self.event_arg_min = EventArgMin()
        self.time = time

    def handle_event(self, event: Event) -> None:
        """
        Resolve this event, e.g. call individual's apply_event() or remove vaccinated individuals, etc.
        """
        # Ring vaccination, removal from self.infecteds, new infecteds, etc.
        raise NotImplementedError()

    @abstractmethod
    def next_individual_level_event(self) -> Event:
        """
        Using Individual.next_event() and any other pertinent information, what would the next event be?
        """
        raise NotImplementedError()

    @abstractmethod
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
        next_individ = self.next_individual_level_event()
        next_pop = self.event_arg_min(self.event_stack)
        next_comb = self.event_arg_min([next_individ, next_pop])

        self.time = next_comb.time
        self.handle_event(next_comb)

    def vaccinate(self, target: Individual, ve: float, rampup: float) -> None:
        """
        Register with population intent to apply "all or nothing" vaccination to target individual.

        Individual will be removed with probability given by VE when vaccine becomes effective (given by rampup).
        """
        # "all or nothing vaccination"
        if np.random.uniform(0, 1) <= ve:
            self.event_stack.append(
                Event(
                    type="recovery",
                    time=self.time + rampup,
                    individual=target,
                )
            )

    @abstractmethod
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
