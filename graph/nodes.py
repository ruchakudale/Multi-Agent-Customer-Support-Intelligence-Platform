from graph.state import TicketState

from agents.intake_agent import intake_agent
from agents.classification_agent import classification_agent
from agents.priority_agent import priority_agent
from agents.retrieval_agent import retrieval_agent
from agents.response_agent import response_agent
from agents.escalation_agent import escalation_agent
from agents.logging_agent import logging_agent


def intake_node(state: TicketState) -> TicketState:
    return intake_agent(state)


def classification_node(state: TicketState) -> TicketState:
    return classification_agent(state)


def priority_node(state: TicketState) -> TicketState:
    return priority_agent(state)


def retrieval_node(state: TicketState) -> TicketState:
    return retrieval_agent(state)


def response_node(state: TicketState) -> TicketState:
    return response_agent(state)


def escalation_node(state: TicketState) -> TicketState:
    return escalation_agent(state)


def logging_node(state: TicketState) -> TicketState:
    return logging_agent(state)