"""
Adapted from PettingZoo (https://github.com/PettingZoo-Team/PettingZoo)
"""

from typing import Any, Generic, TypeVar

AgentT = TypeVar("AgentT")


class AgentSelector(Generic[AgentT]):
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order.

    Example:
        >>> from pettingzoo.utils import AgentSelector
        >>> agent_selector = AgentSelector(agent_order=["player1", "player2"])
        >>> agent_selector.reset()
        'player1'
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        True
        >>> agent_selector.reinit(agent_order=["player2", "player1"])
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        False
    """

    def __init__(self, agent_order: list[AgentT]) -> None:
        self.reinit(agent_order)

    def reinit(self, agent_order: list[AgentT]) -> None:
        """Reinitialize to a new order."""
        self.agent_order: list[AgentT] = agent_order
        self._current_agent: int = 0
        self.selected_agent: AgentT | None = None

    def reset(self) -> AgentT:
        """Reset to the original order."""
        self.reinit(self.agent_order)
        return self.next()

    def next(self) -> AgentT:
        """Get the next agent."""
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self) -> bool:
        """Check if the current agent is the last agent in the cycle."""
        return self.selected_agent == self.agent_order[-1]

    def is_first(self) -> bool:
        """Check if the current agent is the first agent in the cycle."""
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentSelector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )
