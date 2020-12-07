"""
Tests the inference module.
"""

# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=too-many-public-methods

import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from pyugm.factor import DiscreteFactor, DiscreteBelief
from pyugm.infer import Inference
from pyugm.infer_message import LoopyBeliefUpdateInference
from pyugm.infer_message import FloodingProtocol
from pyugm.infer_message import DistributeCollectProtocol
from pyugm.infer_message import LoopyDistributeCollectProtocol
from pyugm.infer_message import multiply
from pyugm.infer_message import ExhaustiveEnumeration
from pyugm.model import Model
from pyugm.tests.test_utils import GraphTestCase


class TestFactorMultiplication(unittest.TestCase):
    def test_multiply_small_inplace(self):
        data = np.array([[1, 2],
                         [5, 6]])
        af = DiscreteFactor([(0, 2), (1, 2)], data=data)
        a = DiscreteBelief(af)

        data = np.array([2, 3])
        bf = DiscreteFactor([(1, 2)], data=data)
        b = DiscreteBelief(bf)

        data = np.array([[2, 6],
                         [10, 18]])
        c = DiscreteFactor([(0, 2), (1, 2)], data=data)

        multiply(a, b)

        print a.data
        print c.data
        print a.data.shape
        print c.data.shape
        print af.data
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_multiply_small_a(self):
        data = np.array([[1, 2],
                         [5, 6]], dtype='float64')
        af = DiscreteFactor([(0, 2), (1, 2)], data=data)
        a = DiscreteBelief(af)

        data = np.array([2, 3])
        e = DiscreteFactor([(0, 2)], data=data)
        data = np.array([[1 * 2, 2 * 2],
                         [5 * 3, 6 * 3]])
        f = DiscreteFactor([(0, 2), (1, 2)], data=data)

        multiply(a, e)

        print 'a', a.data
        print 'e', e.data
        print
        print f.data
        print a.data.shape
        print f.data.shape
        self.assertEqual(a.variables, f.variables)
        self.assertEqual(a.axis_to_variable, f.axis_to_variable)
        assert_array_almost_equal(a.data, f.data)

    def test_multiply_larger(self):
        data = np.array([[[2, 1, 2],
                         [3, 7, 4]],
                        [[1, 1, 3],
                         [4, 9, 10]]])
        af = DiscreteFactor([(0, 2), (3, 2), (12, 3)], data=data)
        a = DiscreteBelief(af)

        data = np.array([[2, 3, 1],
                         [5, 1, 7]])
        b = DiscreteFactor([(0, 2), (12, 3)], data=data)
        data = np.array([[[2 * 2, 1 * 3, 2 * 1],
                          [3 * 2, 7 * 3, 4 * 1]],
                         [[1 * 5, 1 * 1, 3 * 7],
                         [4 * 5, 9 * 1, 10 * 7]]])
        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)], data=data)

        multiply(a, b)

        print a.data
        print c.data
        print a.data.shape
        print c.data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_multiply_larger_correct_order(self):
        data = np.array([[[2, 1, 2],
                          [3, 7, 4]],
                        [[1, 1, 3],
                         [4, 9, 10]]])
        a = DiscreteFactor([(0, 2), (3, 2), (12, 3)], data=data)
        data = np.array([[2, 5],
                         [3, 1],
                         [1, 7]])
        b = DiscreteFactor([(12, 3), (0, 2)], data=data)

        data = np.array([[[2 * 2, 1 * 3, 2 * 1],
                          [3 * 2, 7 * 3, 4 * 1]],
                         [[1 * 5, 1 * 1, 3 * 7],
                          [4 * 5, 9 * 1, 10 * 7]]])
        c = DiscreteFactor([(0, 2), (3, 2), (12, 3)], data=data)

        multiply(a, b)

        print a.data
        print c.data
        print a.data.shape
        print c.data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)

    def test_divide_small(self):
        a = DiscreteFactor([(0, 2), (1, 2)], data=np.array([[1.0, 2], [5, 6]]))
        b = DiscreteFactor([(1, 2)], data=np.array([2.0, 3]))
        data = np.array([[1.0 / 2.0, 2.0 / 3.0],
                         [5.0 / 2.0, 6.0 / 3.0]])
        c = DiscreteFactor([(0, 2), (1, 2)], data=data)
        multiply(a, b, divide=True)

        print a.data
        print c.data
        print a.data.shape
        print c.data.shape
        self.assertEqual(a.variables, c.variables)
        self.assertEqual(a.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(a.data, c.data)


class TestBeliefUpdateInference(GraphTestCase):
    def test_set_up_separators(self):
        a = DiscreteFactor([(0, 2), (1, 2), (2, 2)])
        b = DiscreteFactor([(2, 2), (3, 2), (3, 2)])

        model = Model([a, b])
        inference = LoopyBeliefUpdateInference(model)

        s = DiscreteFactor([(2, 2)])
        print inference._separator_potential
        forward_edge = list(model.edges)[0]
        forward_and_backward_edge = [forward_edge, (forward_edge[1], forward_edge[0])]
        for edge in forward_and_backward_edge:
            separator_factor = inference._separator_potential[edge]

            self.assertSetEqual(separator_factor.variable_set, s.variable_set)
            self.assertDictEqual(separator_factor.cardinalities, s.cardinalities)
            assert_array_almost_equal(separator_factor.data, s.data)

    def test_update_beliefs_small(self):
        a = DiscreteFactor([0, 1])
        b = DiscreteFactor([1, 2])
        model = Model([a, b])
        update_order1 = FloodingProtocol(model=model, max_iterations=2)
        inference = LoopyBeliefUpdateInference(model, update_order1)
        #                       0
        #                     0  1
        # Phi* = Sum_{0} 1 0 [ 1 1 ]  =  1 0 [ 2 ]
        #                  1 [ 1 1 ]       1 [ 2 ]
        #
        #                                        1               1
        # Psi* = Phi* x Psi  =  1 0 [2] x 2 0 [ 1 1 ]  =  2 0 [ 2 2 ]
        #        Phi              1 [2]     1 [ 1 1 ]       1 [ 2 2 ]
        #
        #                        1           1
        # Phi** = Sum_{2} 2 0 [ 2 2 ]  =  [ 4 4 ]
        #                   1 [ 2 2 ]
        #
        #                            1              0               0
        # Psi** = Phi** x Psi  =  [ 2 2 ] x  1 0 [ 1 1 ]  =  1 0 [ 2 2 ]
        #         Phi*                         1 [ 1 1 ]       1 [ 2 2 ]
        #
        #             1
        # Phi*** = [ 4 4 ]
        #                                 1
        # Psi*** = Phi*** x Psi* = 2 0 [ 2 2 ]
        #          Phi**             1 [ 2 2 ]
        #
        inference.calibrate()
        #update_order2 = FloodingProtocol(model=model, max_iterations=3)
        #change1, iterations1 = inference.calibrate(update_order2)
        #print 'changes:', change0, change1, 'iterations:', iterations0, iterations1

        final_a_data = np.array([[2, 2],
                                 [2, 2]], dtype='f64') / 8.0
        final_b_data = np.array([[2, 2],
                                 [2, 2]], dtype='f64') / 8.0

        belief_a = inference.beliefs[a]
        assert_array_almost_equal(final_a_data, belief_a.normalized_data)
        belief_b = inference.beliefs[b]
        assert_array_almost_equal(final_b_data, belief_b.normalized_data)

    def test_update_beliefs_disconnected(self):
        a = DiscreteFactor([(1, 2), (2, 2)], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        b = DiscreteFactor([(2, 2), (3, 2)], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        c = DiscreteFactor([(4, 2), (5, 2)], data=np.array([[5, 6], [8, 9]], dtype=np.float64))
        d = DiscreteFactor([(5, 2), (6, 2)], data=np.array([[1, 6], [2, 3]], dtype=np.float64))
        e = DiscreteFactor([(7, 2), (8, 2)], data=np.array([[2, 1], [2, 3]], dtype=np.float64))

        model = Model([a, b, c, d, e])
        for factor in model.factors:
            print 'before', factor, np.sum(factor.data)

        update_order = DistributeCollectProtocol(model)
        inference = LoopyBeliefUpdateInference(model, update_order=update_order)

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate().belief
        print 'Exhaust', np.sum(exhaustive_answer.data)

        change = inference.calibrate()
        print change

        for factor in model.factors:
            print factor, np.sum(factor.data)

        for variable in model.variables:
            marginal_beliefs = inference.get_marginals(variable)
            true_marginal = exhaustive_answer.marginalize([variable])
            for marginal in marginal_beliefs:
                assert_array_almost_equal(true_marginal.normalized_data, marginal.normalized_data)

        expected_ln_Z = np.log(exhaustive_answer.data.sum())
        self.assertAlmostEqual(expected_ln_Z, inference.partition_approximation())

    def test_belief_update_larger_tree(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 3], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        d = DiscreteFactor([3], data=np.array([2, 1], dtype=np.float64))
        e = DiscreteFactor([0], data=np.array([4, 1], dtype=np.float64))
        f = DiscreteFactor([2], data=np.array([1, 2], dtype=np.float64))
        #
        # a{0 1} - b{1 2} - c{2 3} - d{3}
        #    |       |
        # e{0}     f{2}
        #
        model = Model([a, b, c, d, e, f])
        print 'edges', model.edges
        update_order = DistributeCollectProtocol(model)
        inference = LoopyBeliefUpdateInference(model, update_order=update_order)

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate().belief

        print 'bp'
        change = inference.calibrate()
        print change

        for factor in model.factors:
            print factor

        for variable in model.variables:
            marginal_beliefs = inference.get_marginals(variable)
            true_marginal = exhaustive_answer.marginalize([variable])
            for marginal in marginal_beliefs:
                assert_array_almost_equal(true_marginal.normalized_data, marginal.normalized_data)

        expected_ln_Z = np.log(exhaustive_answer.data.sum())
        self.assertAlmostEqual(expected_ln_Z, inference.partition_approximation())

    def test_belief_update_long_tree(self):
        label_template = np.array([['same', 'different'],
                                   ['different', 'same']])
        observation_template = np.array([['obs_low'] * 32,
                                         ['obs_high'] * 32])
        observation_template[0, 13:17] = 'obs_high'
        observation_template[1, 13:17] = 'obs_low'
        N = 2
        pairs = [DiscreteFactor([(i, 2), (i + 1, 2)], parameters=label_template) for i in xrange(N - 1)]
        obs = [DiscreteFactor([(i, 2), (i + N, 32)], parameters=observation_template) for i in xrange(N)]
        repe = [16., 16., 14., 13., 15., 16., 14., 13., 15., 16., 15.,
                13., 14., 16., 16., 15., 13., 13., 14., 14., 13., 14.,
                14., 14., 14., 14., 14., 14., 14., 14., 14., 14., 14.,
                14., 14., 14., 14., 14., 14., 14., 14., 9., 4., 4.,
                4., 4., 5., 3., 2., 3., 2., 3., 3., 3., 3.,
                3., 3., 3., 3., 4., 4., 5., 5., 5.]
        evidence = dict((i + N, 0 if repe[i % len(repe)] >= 13 and repe[i % len(repe)] < 17 else 1) for i in xrange(N))

        model = Model(pairs + obs)
        parameters = {'same': 2.0, 'different': -1.0, 'obs_high': 0.0, 'obs_low': -0.0}

        update_order = FloodingProtocol(model, max_iterations=4)
        inference = LoopyBeliefUpdateInference(model, update_order=update_order)
        inference.calibrate(evidence, parameters)

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate(evidence, parameters).belief

        for i in xrange(N):
            expected_marginal = exhaustive_answer.marginalize([i])
            for actual_marginal in inference.get_marginals(i):
                print i
                print expected_marginal.normalized_data
                print actual_marginal.normalized_data
                assert_array_almost_equal(expected_marginal.normalized_data, actual_marginal.normalized_data)

        expected_ln_Z = np.log(exhaustive_answer.data.sum())
        self.assertAlmostEqual(expected_ln_Z, inference.partition_approximation())


class TestLoopyBeliefUpdateInference(GraphTestCase):
    def test_loopy_distribute_collect(self):
        a = DiscreteFactor([0, 1], data=np.array([[1, 2], [2, 2]], dtype=np.float64))
        b = DiscreteFactor([1, 2], data=np.array([[3, 2], [1, 2]], dtype=np.float64))
        c = DiscreteFactor([2, 0], data=np.array([[1, 2], [3, 4]], dtype=np.float64))
        #
        # a{0 1} - b{1 2}
        #    \       /
        #      c{2 0}
        #
        # a{0 1} - {0} - c{2 0}
        #
        #
        #
        #
        model = Model([a, b, c])
        update_order = LoopyDistributeCollectProtocol(model, max_iterations=40)
        inference = LoopyBeliefUpdateInference(model, update_order=update_order)
        inference.calibrate()

        exact_inference = ExhaustiveEnumeration(model)
        exhaustive_answer = exact_inference.calibrate().belief

        for factor in model.factors:
            print factor, np.sum(factor.data)
        for var in model.variables_to_factors.keys():
            print var, exhaustive_answer.marginalize([var]).data
        print
        for var in model.variables_to_factors.keys():
            print var, inference.get_marginals(var)[0].data

        for variable in model.variables:
            for factor in inference.get_marginals(variable):
                expected_table = exhaustive_answer.marginalize([variable])
                actual_table = factor.marginalize([variable])
                assert_array_almost_equal(expected_table.normalized_data, actual_table.normalized_data, decimal=2)

        expected_ln_Z = np.log(exhaustive_answer.data.sum())
        self.assertAlmostEqual(expected_ln_Z, inference.partition_approximation(), places=1)

    def test_loopy_distribute_collect_grid(self):
        a = DiscreteFactor([0, 1], data=np.random.randn(2, 2))
        b = DiscreteFactor([1, 2, 3])
        c = DiscreteFactor([3, 4, 5])
        d = DiscreteFactor([5, 6])
        e = DiscreteFactor([0, 7, 8])
        f = DiscreteFactor([8, 2, 9, 10])
        g = DiscreteFactor([10, 4, 11, 12])
        h = DiscreteFactor([12, 13, 6])
        i = DiscreteFactor([7, 14])
        j = DiscreteFactor([14, 9, 15])
        k = DiscreteFactor([15, 11, 16])
        l = DiscreteFactor([16, 13])

        # a{0 1} ---[1]--- b{1 2 3} ---[3]--- c{3 4 5} ---[5]--- d{5 6}
        #   |                  |                   |                  |
        #  [0]                [2]                 [4]                [6]
        #   |                  |                   |                  |
        # e{0 7 8} --[8]-- f{8 2 9 10} --[10]- g{10 4 11 12} -[12]- h{12 13 6}
        #   |                  |                   |                  |
        #  [7]                [9]                 [11]               [13]
        #   |                  |                   |                  |
        # i{7 14} --[14]--j{14 9 15} --[15]-- k{15 11 16} --[16]-- l{16 13}

        model = Model([a, b, c, d, e, f, g, h, i, j, k, l])

    def test_exhaustive_enumeration(self):
        a = DiscreteFactor([(0, 2), (1, 3)], data=np.array([[1, 2, 3], [4, 5, 6]]))
        b = DiscreteFactor([(0, 2), (2, 2)], data=np.array([[1, 2], [2, 1]]))
        # 0 1 2 |
        #-------+--------
        # 0 0 0 | 1x1=1
        # 0 0 1 | 1x2=2
        # 0 1 0 | 2x1=2
        # 0 1 1 | 2x2=4
        # 0 2 0 | 3x1=3
        # 0 2 1 | 3x2=6
        # 1 0 0 | 4x2=8
        # 1 0 1 | 4x1=4
        # 1 1 0 | 5x2=10
        # 1 1 1 | 5x1=5
        # 1 2 0 | 6x2=12
        # 1 2 1 | 6x1=6

        model = Model([a, b])
        exact_inference = ExhaustiveEnumeration(model)
        c = exact_inference.calibrate().belief

        d = DiscreteFactor([(0, 2), (1, 3), (2, 2)])
        d._data = np.array([1, 2, 2, 4, 3, 6, 8, 4, 10, 5, 12, 6]).reshape(2, 3, 2)

        self.assertEqual(d.variables, c.variables)
        self.assertEqual(d.axis_to_variable, c.axis_to_variable)
        assert_array_almost_equal(d._data, c.data)


if __name__ == '__main__':
    unittest.main()
