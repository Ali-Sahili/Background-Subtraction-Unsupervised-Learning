"""
Module containing the Model class.
"""
# License: BSD 3 clause

import numpy


class Model(object):
    """
    Class to contain the factors and their relationships.
    """

    def __init__(self, factor_list):
        """
        Constructor.
        :param factor_list: List of factors to add to this model.
        """
        self.factors = []  # factor objects
        self.cardinalities = dict()  # variable name to int
        self.edges = set()  # pairs of factors
        self.disconnected_subgraphs = []  # list of sets of factors
        self.variables_to_factors = dict()  # variable name to factor

        for factor in factor_list:
            self._add_factor(factor)
        
        self._build_graph()

    def _add_factor(self, factor):
        """
        Helper to add a factor to the model.
        :param factor: Factor to add.
        """
        self.factors.append(factor)

        for variable in factor.variables:
            if variable[0] in self.cardinalities:
                assert(variable[1] == self.cardinalities[variable[0]])
            else:
                self.cardinalities[variable[0]] = variable[1]

            if variable[0] in self.variables_to_factors:
                self.variables_to_factors[variable[0]].add(factor)
            else:
                self.variables_to_factors[variable[0]] = {factor}

    def _build_graph(self):
        """
        Helper to build a cluster graph given the factors.

        Greedily add the edge connecting the two unconnected factors that share the most variables.
        """

        def _add_edge_to_set(edge, set_to_add):
            """
            Helper to add an edge to a set - the edge and its inverse must be added.
            :param edge: The edge to add.
            :param set_to_add: The set to add the edge to.
            """
            # Edge should be undirected - but we have to use tuples (need to be hashable)
            if (edge[1], edge[0]) not in set_to_add:
                set_to_add.add(edge)

        def _mark_factors_in_edge(edge, marked_set, unmarked_set):
            """
            Remove an edge from one set and add it to another.
            :param edge: The edge to move.
            :param marked_set: The set to remove from.
            :param unmarked_set: The set to add to.
            """
            marked_set.add(edge[0])
            marked_set.add(edge[1])
            if edge[0] in unmarked_set:
                unmarked_set.remove(edge[0])
            if edge[1] in unmarked_set:
                unmarked_set.remove(edge[1])

        print('building graph')
        # Build graph by greedily adding the largest sepset factors to the above added node
        for variable, factors in self.variables_to_factors.items():
            #print('for loop')
            marked_factors = set()
            unmarked_factors = set(factors)

            if len(factors) > 1:
                first_candidate_sepset = self._get_largest_unmarked_sepset(variable,
                                                                           list(factors),
                                                                           unmarked_factors,  # just for start
                                                                           unmarked_factors)
                _add_edge_to_set(first_candidate_sepset, self.edges)
                _mark_factors_in_edge(first_candidate_sepset, marked_factors, unmarked_factors)

                while len(marked_factors) < len(factors):
                    #print('while loop')
                    largest_sepset = self._get_largest_unmarked_sepset(variable, list(factors), marked_factors,unmarked_factors)
                    # Add largest sepset factors to graph
                    if largest_sepset is not None:
                        _add_edge_to_set(largest_sepset, self.edges)
                        _mark_factors_in_edge(largest_sepset, marked_factors, unmarked_factors)

        print('disconnecting...')
        self._find_disconnected_subgraphs()

    @staticmethod
    def _get_largest_unmarked_sepset(variable, factors, marked_factors, unmarked_factors):
        """
        Find the edge connecting two unmarked factors with the largest separator set.
        :param variable: A variable that the two factors must share.
        :param factors: List of all factors.
        :param marked_factors: Set of marked factors - one of the factors in the edge must be in this set.
        :param unmarked_factors: Set of unmarked factors - the other factor in the edge must be in this set.
        """
        sepset_sizes = [((factor1, factor2), len(factor1.variable_set.intersection(factor2.variable_set)))
                        for factor1 in factors for factor2 in factors
                        if (factor1 in marked_factors
                            and factor2 in unmarked_factors
                            and factor1 != factor2
                            and variable in factor1.variable_set
                            and variable in factor2.variable_set)]
        if len(sepset_sizes) == 0:
            max_sepset = (None, 0)
        else:
            max_sepset = max(sepset_sizes, key=lambda x: x[1])
        return max_sepset[0]

    def _find_disconnected_subgraphs(self):
        """
        Helper to find islands of factor nodes and add them to the `disconnected_subgraphs` attribute.
        """
        def _connected_factors(factor_to_follow):
            """
            Helper to find the set of factors connected to a certain factor.
            :param factor_to_follow: The factor.
            :returns: Set of factors.
            """
            return_set = set()
            for edge in self.edges:
                if factor_to_follow in edge:
                    return_set.add(edge[0] if edge[1] == factor_to_follow else edge[1])
            return return_set

        visited_factors = set()
        kkk = 0
        for factor in self.factors:
            print('----',kkk,'----')
            kkk += 1
            new_set = set()
            if factor not in visited_factors:
                new_set.add(factor)
                visited_factors.add(factor)
                factors_to_visit = _connected_factors(factor)
                if len(factors_to_visit) > 0:
                    next_factor = factors_to_visit.pop()
                    while next_factor:
                        new_set.add(next_factor)
                        visited_factors.add(next_factor)
                        factors_to_visit = factors_to_visit.union(_connected_factors(next_factor).difference(visited_factors))
                        try:
                            next_factor = factors_to_visit.pop()
                        except KeyError:
                            next_factor = None
                self.disconnected_subgraphs.append(new_set)

    @property
    def variables(self):
        """
        The list of variables present in the model.
        """
        return [key for key, _ in self.variables_to_factors.items()]
