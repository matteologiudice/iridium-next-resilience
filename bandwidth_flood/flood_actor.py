"""Traffic-generating actor used by the distributed bandwidth exhaustion scenario.

This is a local variant of ``dsns.traffic_sim.TrafficFloodActor``. The simulator's own actor is
left untouched; it is not used here because it emits ``HybridDirectMessage`` objects, which the
routing actor always dispatches down the LTP path regardless of the configured delivery mode.
Without a dedicated retransmission actor, which this work never uses, those messages are created
but never sent. This variant keeps the same logic (links, period, size, start_time, end_time, and
the same ``TrafficFloodEvent``) and emits an ordinary ``DirectMessage`` instead.
"""

from dsns.events import Event
from dsns.message import DirectMessage, MessageCreatedEvent
from dsns.simulation import Actor
from dsns.traffic_sim import TrafficFloodEvent


class SimpleFloodActor(Actor):
    """Periodically creates a message on each of the given links while the attack is active."""

    def __init__(self, links: list[tuple[int, int]], period: float = 1.0, size: int = None,
                 start_time: float = 0.0, end_time: float = float("inf")):
        super().__init__()
        self.links = links
        self.period = period
        self.size = size
        self.start_time = start_time
        self.end_time = end_time

    def initialize(self) -> list[Event]:
        return [TrafficFloodEvent(self.start_time)]

    def _generate_events(self, time: float) -> list[Event]:
        if time <= self.end_time:
            events = [
                MessageCreatedEvent(
                    time=time,
                    message=DirectMessage(
                        source=source,
                        destination=destination,
                        data=f"Flooding message: {source} -> {destination}",
                        size=self.size,
                    ),
                )
                for (source, destination) in self.links
            ]
            events.append(TrafficFloodEvent(time=time + self.period))
            return events
        return []

    def handle_event(self, _, event: Event) -> list[Event]:
        if isinstance(event, TrafficFloodEvent):
            return self._generate_events(event.time)
        return []
