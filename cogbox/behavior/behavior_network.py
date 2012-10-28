#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Implementation of the a Behavior Network based on Maes, P. (1989)
How To Do The Right Thing
"""
from __future__ import division

__author__ = "Anderson Vieira"


class Behavior(object):
    """
    Public methods:
    execute -- Stores a pattern in the memory according to a given address

    Instance variables:
    label -- label of the behavior
    action -- action that runs when the behavior is executed
    preconditions -- preconditions necessary for the behavior to execute
    additions -- items added to the state when the behavior is executed
    deletions -- items deleted from the state when the behavior is executed
    previous_activation -- activation in the step before the current step
    current_activation -- activation in this step
    executable -- true when the behavior can be executed
    """

    def __init__(self, label, action, preconditions=frozenset(),
                 additions=frozenset(), deletions=frozenset()):
        """
        :param label: label of the behavior
        :param action: action that runs when the behavior is executed
        :param preconditions: preconditions necessary for the
        behavior to execute
        :param additions: items added to the world state when the behavior
        is executed
        :param deletions: items deleted from the world state when the
        behavior is executed
        :type label: string
        :type action: action
        :type preconditions: set
        :type additions: set
        :type deletions: set
        """
        self.label = label
        self.action = action
        self.preconditions = preconditions
        self.additions = additions
        self.deletions = deletions
        self.previous_activation = 0.0
        self.current_activation = 0.0
        self.executable = False

    def __str__(self):
        """
        Return the label of the behavior.
        """
        return "%s" % (self.label)

    def __repr__(self):
        """
        Return the label of the behavior when printing a list.
        """
        return self.__str__()

    def __gt__(self, other):
        if other is None:
            return True
        else:
            return self.current_activation > other.current_activation

    def __lt__(self, other):
        if other is None:
            return False
        else:
            return self.current_activation < other.current_activation

    def __ge__(self, other):
        if other is None:
            return True
        else:
            return self.current_activation >= other.current_activation

    def __le__(self, other):
        if other is None:
            return False
        else:
            return self.current_activation <= other.current_activation

    def execute(self):
        """
        Resets the current activation level of the behavior and executes
        the action associated with it.
        """
        assert self.executable is True
        self.current_activation = 0
        self.action.execute()


class Energy(object):
    """
    Energy levels used through the network.

    Instance variables:
    data -- pushes the network towards exploring available data
    goals -- pushes the networks towards accomplishing the goals
    conf -- pushes the network towards not undoing protected goals
    mean -- mean energy that must be maintained accross iterations
    """
    def __init__(self, data_energy=20., goals_energy=70., conf_energy=50.,
                 mean_energy=20.):
        self.data = data_energy
        self.goals = goals_energy
        self.conf = conf_energy
        self.mean = mean_energy


class State(object):
    """
    State of the world.

    Instance variables:
    data -- set of items that describe the current state of the world
    goals -- the goals of the agent
    protected_goals -- the protected goals that were accomplished
    """
    def __init__(self, data=frozenset(), goals=frozenset(),
                 protected_goals=frozenset()):
        self.data = data
        self.goals = goals
        self.protected_goals = protected_goals


class Network(object):
    """
    Behvaior Network

    Instance variables:
    behaviors -- list of behaviors that compose the network
    data -- state of the world that the network has access to
    goals -- goals that the network tries to achieve
    protected_goals -- goals that once achieved should not be turned off
    max_threshold -- max threshold value for activating a behavior
    threshold -- current value for activating a behavior
    data_energy -- pushes the network towards exploring available data
    goal_energy -- pushes the network towards accomplishing the goals
    conf_energy -- pushes the network towards not undoing protected goals
    mean_energy -- mean energy that must be maintained accross iterations
    """
    def __init__(self, behaviors, energy=Energy(), max_threshold=45.):
        self.behaviors = behaviors
        self.threshold = self.max_threshold = max_threshold
        self.energy = energy

    def behaviors_that_need(self, item):
        """
        Return the list of behaviors that have an item in their
        preconditions.

        :param item: item we are searching
        :type item: string
        :rtype: list

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, set(["foo"]), set(), set())
        >>> b2 = bn.Behavior("b2", None, set(["bar"]), set(), set())
        >>> net = bn.Network([b1, b2])
        >>> net.behaviors_that_need("foo")
        [b1]
        """
        return [behavior for behavior in self.behaviors
                if item in behavior.preconditions]

    def behaviors_that_add(self, item):
        """
        Return the list of behaviors that have an item in their
        additions.

        :param item: item we are searching
        :type item: string
        :rtype: list

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, set(), set(["foo"]), set())
        >>> b2 = bn.Behavior("b2", None, set(), set(["bar"]), set())
        >>> net = bn.Network([b1, b2])
        >>> net.behaviors_that_add("foo")
        [b1]
        """
        return [behavior for behavior in self.behaviors
                if item in behavior.additions]

    def behaviors_that_delete(self, item):
        """
        Return the list of behaviors that have an item in their
        deletions.

        :param item: item we are searching
        :type item: string
        :rtype: list

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, set(), set(), set(["foo"]))
        >>> b2 = bn.Behavior("b2", None, set(), set(), set(["bar"]))
        >>> net = bn.Network([b1, b2])
        >>> net.behaviors_that_delete("foo")
        [b1]
        """
        return [behavior for behavior in self.behaviors
                if item in behavior.deletions]

    def mean_activation(self):
        """
        Return the current mean level of behavior activation.
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, set(), set(), set())
        >>> b2 = bn.Behavior("b2", None, set(), set(), set())
        >>> net = bn.Network([b1, b2])
        >>> net.mean_activation()
        0.0
        >>> b1.current_activation = 2.0
        >>> b2.current_activation = 4.0
        >>> net.mean_activation()
        3.0
        """
        return (sum([behavior.current_activation for behavior
                     in self.behaviors]) /
                len(self.behaviors))

    def input_from_data(self, behavior, state):
        """
        Return the input energy that a given behavior gets from data.
        A behavior gets energy from a data item when it has that item
        in its preconditions.

        :param behavior: behavior that will receive the energy
        :type behavior: Behavior
        :return: the input energy that a given behavior gets from data.
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, preconditions=set(["foo"]))
        >>> b2 = bn.Behavior("b2", None, preconditions=set(["bar"]))
        >>> net = bn.Network([b1, b2])
        >>> state = State(data=set(["foo"]))
        >>> net.input_from_data(b1, state) == net.energy.data
        True
        >>> b3 = bn.Behavior("b3", None, preconditions=set(["foo"]))
        >>> net.behaviors.append(b3)
        >>> net.input_from_data(b1, state) == net.energy.data / 2
        True
        """
        return (self.energy.data *
                sum([(1. / len(self.behaviors_that_need(item))) *
                     (1. / len(behavior.preconditions))
                     for item in
                     (state.data | state.protected_goals) &
                     behavior.preconditions]))

    def input_from_goals(self, behavior, state):
        """
        Return the input energy that a given behavior gets from goals.
        A behavior will get energy from a goal when it has that goal
        in its additions.

        :param behavior: behavior that will receive the energy
        :type behavior: Behavior
        :return: the input energy that a given behavior gets from goals.
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, additions=set(["foo"]))
        >>> b2 = bn.Behavior("b2", None, additions=set(["bar"]))
        >>> net = bn.Network([b1, b2])
        >>> state = State(goals=set(["foo"]))
        >>> net.input_from_goals(b1, state) == net.energy.goals
        True
        >>> b3 = bn.Behavior("b3", None, additions=set(["foo"]))
        >>> net.behaviors.append(b3)
        >>> net.input_from_goals(b1, state) == net.energy.goals / 2
        True
        """
        return (self.energy.goals *
                sum([(1. / len(self.behaviors_that_add(item))) *
                     (1. / len(behavior.additions))
                     for item in
                     (state.goals & behavior.additions)]))

    def taken_by_protected_goals(self, behavior, state):
        """
        Return the amount energy that a given behavior loses because
        of protected goals. A protected goal will take energy away from
        a behavior when that behavior has the protected goal in its
        deletions.

        :param behavior: behavior that will receive the energy
        :type behavior: Behavior
        :return: the amount of energy that is taken away from the behavior
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> b1 = bn.Behavior("b1", None, deletions=set(["foo"]))
        >>> b2 = bn.Behavior("b2", None, deletions=set(["bar"]))
        >>> net = bn.Network([b1, b2])
        >>> state = State(protected_goals=set(["foo"]))
        >>> net.taken_by_protected_goals(b1, state) == net.energy.conf
        True
        >>> b3 = bn.Behavior("b3", None, deletions=set(["foo"]))
        >>> net.behaviors.append(b3)
        >>> net.taken_by_protected_goals(b1, state) == net.energy.conf / 2
        True
        """
        return (self.energy.conf *
                sum([(1. / len(self.behaviors_that_delete(item))) *
                     (1. / len(behavior.deletions))
                     for item in
                     (state.protected_goals & behavior.deletions)]))

    def spread_backwards(self, source, destination, state):
        """
        Return the amount of energy that the source behavior spreads
        backwards to the destination behavior. A behavior will spread
        energy backwards to another behavior when the first has an item
        in its preconditions and the second has the same item in its
        additions. A behavior will give away its activation energy only
        if its not currently executable.

        :param source: behavior that is giving energy
        :param destination: behavior that is receiving energy
        :return: amount of energy from source to destination
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> source = bn.Behavior("b1", None, preconditions=set(["foo"]))
        >>> good_dest = bn.Behavior("b2", None, additions=set(["foo"]))
        >>> bad_dest = bn.Behavior("b3", None, additions=set(["bar"]))
        >>> net = bn.Network([source, good_dest, bad_dest])
        >>> state = State()
        >>> net.spread_backwards(source, good_dest, state)
        0.0
        >>> source.previous_activation = 5.0
        >>> net.spread_backwards(source, good_dest, state)
        5.0
        >>> net.spread_backwards(source, bad_dest, state)
        0.0
        >>> source.executable = True
        >>> net.spread_backwards(source, good_dest, state)
        0.0
        """
        if source.executable:
            return 0.0
        else:
            return (source.previous_activation *
                    sum([(1. / len(self.behaviors_that_add(item))) *
                         (1. / len(destination.additions))
                         for item in
                         ((source.preconditions & destination.additions) -
                          state.data)]))

    def spread_forward(self, source, destination, state):
        """
        Return the amount of energy that the source behavior spreads
        forward to the destination behavior. A behavior will spread
        energy forward to another behavior when the first has an item
        in its additions and the second has the same item in its
        preconditions. A behavior will give away its activation energy only
        if its not currently executable.

        :param source: behavior that is giving energy
        :param destination: behavior that is receiving energy
        :return: amount of energy from source to destination
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> source = bn.Behavior("b1", None, additions=set(["foo"]))
        >>> good_dest = bn.Behavior("b2", None, preconditions=set(["foo"]))
        >>> bad_dest = bn.Behavior("b3", None, preconditions=set(["bar"]))
        >>> net = bn.Network([source, good_dest, bad_dest])
        >>> state = State()
        >>> net.spread_forward(source, good_dest, state)
        0.0
        >>> source.previous_activation = net.energy.goals
        >>> net.spread_forward(source, good_dest, state) == net.energy.data
        True
        >>> net.spread_forward(source, bad_dest, state)
        0.0
        >>> source.executable = True
        >>> net.spread_forward(source, good_dest, state)
        0.0
        """
        if source.executable:
            return 0.0
        else:
            return (source.previous_activation *
                    (self.energy.data / self.energy.goals) *
                    sum([(1. / len(self.behaviors_that_need(item))) *
                         (1. / len(destination.preconditions))
                         for item in
                         ((source.additions & destination.preconditions) -
                          state.data)]))

    def take_away(self, taker, source, state):
        """
        Return the amount of energy that the taker behavior takes
        away from the source behavior. A behavior will take energy
        away from another behavior when the first has an item
        in its preconditions and the second has the same item in its
        deletions. When this happens it is said that there is a
        conflictor link between the taker and the source.

        :param taker: behavior that is taking energy
        :param source: behavior that is givingg energy
        :return: amount of energy from source to taker
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> taker = bn.Behavior("b1", None, preconditions=set(["foo"]))
        >>> source_conf = bn.Behavior("b2", None, deletions=set(["foo"]))
        >>> source_not = bn.Behavior("b3", None, deletions=set(["bar"]))
        >>> net = bn.Network([taker, source_conf, source_not])
        >>> state = State(data=set(["foo"]))
        >>> net.take_away(taker, source_conf, state)
        0.0
        >>> taker.previous_activation = net.energy.goals
        >>> net.take_away(taker, source_conf, state) == net.energy.conf
        True
        >>> net.take_away(taker, source_not, state)
        0.0
        >>> state = State()
        >>> net.take_away(taker, source_conf, state)
        0.0
        """
        if ((taker.previous_activation < source.previous_activation) and
                len(state.data & source.preconditions & taker.deletions) > 0):
            return 0.0
        else:
            return (taker.previous_activation *
                    (self.energy.conf / self.energy.goals) *
                    sum([(1. / len(self.behaviors_that_delete(item))) *
                         (1. / len(source.deletions))
                         for item in
                         taker.preconditions & source.deletions &
                         state.data]))

    def behavior_spread(self, target, state):
        """
        Return the total amount of activation energy that the
        target behavior gets from the other behaviors.

        :param destination: behavior that will receive energy
        :type destination: Behavior
        :returns: total amount of activation received by destination
        :rtype: float

        :Example:
        >>> import behavior_network as bn
        >>> target = bn.Behavior("t0", None, set(["pre"]), set(["add"]), \
                set(["del"]))
        >>> b1 = bn.Behavior("b1", None, preconditions=set(["add"]))
        >>> b2 = bn.Behavior("b2", None, additions=set(["pre"]))
        >>> b3 = bn.Behavior("b3", None, preconditions=set(["del"]))
        >>> target.previous_activation = 5.0
        >>> b1.previous_activation = 10.0
        >>> b2.previous_activation = 10.0
        >>> b3.previous_activation = 10.0
        >>> net = Network([target, b1, b2, b3])
        >>> state = State(data=set(["del"]))
        >>> backwards = net.spread_backwards(b1, target, state)
        >>> forward = net.spread_forward(b2, target, state)
        >>> taken = net.take_away(b3, target, state)
        >>> total = backwards + forward - taken
        >>> net.behavior_spread(target, state) == total
        True
        """
        return sum([(self.spread_backwards(behavior, target, state) +
                     self.spread_forward(behavior, target, state) -
                     self.take_away(behavior, target, state))
                    for behavior in self.behaviors if behavior != target])

    def relax(self):
        """
        Lower the total activation of the network so that the mean
        activation is less than or equal to the value given by
        self.energy.mean. Also set the value of the previous activation
        to be equal to the current activation for each behavior.

        :Example:
        >>> import behavior_network as bn
        >>> b0 = bn.Behavior("b0", None)
        >>> b1 = bn.Behavior("b1", None)
        >>> b2 = bn.Behavior("b2", None)
        >>> b3 = bn.Behavior("b3", None)
        >>> net = Network([b0, b1, b2, b3])
        >>> b0.current_activation = 20.0
        >>> b1.current_activation = 25.0
        >>> b2.current_activation = 15.0
        >>> b3.current_activation = 10.0
        >>> net.mean_activation()
        17.5
        >>> b2.current_activation = 40.0
        >>> b3.current_activation = 35.0
        >>> net.mean_activation() <= net.energy.mean
        False
        >>> net.relax()
        >>> net.mean_activation() <= net.energy.mean
        True
        """
        current_mean_activation = self.mean_activation()
        if current_mean_activation > self.energy.mean:
            for behavior in self.behaviors:
                behavior.current_activation *= (self.energy.mean /
                                                current_mean_activation)
                behavior.previous_activation = behavior.current_activation

    def update_behaviors(self, state):
        """
        Check which behaviors are executable and set their executable
        attribute accordingly. Update the behaviors activation energies.
        """
        for behavior in self.behaviors:
            behavior.executable = (behavior.preconditions <=
                                   (state.data | state.protected_goals))
            behavior.current_activation = max(
                ((0.9 * behavior.previous_activation) +
                 self.input_from_data(behavior, state) +
                 self.input_from_goals(behavior, state) -
                 self.taken_by_protected_goals(behavior, state) +
                 self.behavior_spread(behavior, state)),
                0)
        self.relax()

    def active(self, behavior):
        """
        Return true if a behavior should be active regardless of the
        other behaviors in the network.

        :param behavior: behavior to check
        :type behavior: Behavior
        :return: true of a behavior should be active
        :rtype: bool

        :Example:
        >>> import behavior_network as bn
        >>> b0 = bn.Behavior("b0", None, preconditions=set(["foo"]))
        >>> b1 = bn.Behavior("b1", None, preconditions=set(["bar"]))
        >>> b2 = bn.Behavior("b2", None, preconditions=set(["foo"]))
        >>> b3 = bn.Behavior("b3", None, preconditions=set())
        >>> net = Network([b0, b1, b2, b3])
        >>> net.threshold = 20.0
        >>> net.update_behaviors(State(data=set(["foo"])))
        >>> b0.current_activation = 25.0
        >>> b1.current_activation = 25.0
        >>> b2.current_activation = 15.0
        >>> b3.current_activation = 25.0
        >>> net.active(b0)
        True
        >>> net.active(b1)
        False
        >>> net.active(b2)
        False
        >>> net.active(b3)
        True
        """
        return (behavior.executable and
                behavior.current_activation >= self.threshold)

    def active_behavior(self):
        """
        Search for a behavior that satisfies the three conditions below:
        i) the behavior is executable (all its preconditions are satisfied)
        ii) the behavior activation is greater than the threshold
        iii) the behavior has the greates activation among those that
             satisfy (i) and (ii)
        If no such behavior is found, return None

        :Example:
        >>> import behavior_network as bn
        >>> b0 = bn.Behavior("b0", None, set(["foo"]), set(["bar"]))
        >>> b1 = bn.Behavior("b1", None, preconditions=set(["bar"]))
        >>> b2 = bn.Behavior("b2", None, preconditions=set(["foo"]))
        >>> b3 = bn.Behavior("b3", None, set(["bar"]), set(["bar"]))
        >>> net = Network([b0, b1, b2, b3])
        >>> net.threshold = 5.0
        >>> b0.current_activation = 15.0
        >>> b1.current_activation = 15.0
        >>> b2.current_activation = 15.0
        >>> b3.current_activation = 15.0
        >>> net.update_behaviors(State(data=set(["foo"]), goals=set(["bar"])))
        >>> net.active_behavior()
        b0
        >>> net.update_behaviors(State())
        >>> net.active_behavior() is None
        True
        """
        return max([behavior for behavior in self.behaviors
                    if self.active(behavior)] + [None])

    def run(self, state):
        """
        Call update_behaviors() to update the activavion and status of
        the behaviors.
        Call active_behavior() to get an active behavior.
        If active_behavior() returns None, lower the threshold by 10%
        else, call behavior.act() and set the threshold back to max
        """
        self.update_behaviors(state)
        behavior = self.active_behavior()
        if behavior is None:
            self.threshold *= 0.9
        else:
            behavior.execute()
            self.threshold = self.max_threshold


if __name__ == "__main__":
    import doctest
    doctest.testmod()
