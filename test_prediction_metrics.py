# Unit testing test
# authors: Randal J. Barnes and Elizabeth A. Barnes
# date: September 2, 2022

import numpy as np
import prediction_metrics
import unittest


# def test_correct_prediction_errors():
#
#     # test for when predictions and labels are equal
#     y_data = np.ones((100,5))
#     y_data[:,-1] = 0.
#     label_data = np.ones((100,2))
#     rmse, rmse_cons = prediction_metrics.get_errors(y_data, label_data)
#
#     assert np.all(rmse == 0.0), "perfect predictions should have zero rmse"
#
# def test_incorrect_prediction_errors():
#     # test for when predictions and labels are notequal
#     y_data = np.ones((100,5))
#     y_data[:,-1] = 0.
#     label_data = 3.*np.ones((100,2))
#     rmse, rmse_cons = prediction_metrics.get_errors(y_data, label_data)
#
#     assert np.all(rmse != 0.0), "should get nonzero errors for wrong predictions"


# Using the unittest class
class test_get_errors_unittest(unittest.TestCase):
    def test_correct_prediction_errors_unittest(self):
        # test for when predictions and labels are equal
        y_data = np.ones((100, 5))
        y_data[:, -1] = 0.
        label_data = np.ones((100, 2))
        rmse, rmse_cons = prediction_metrics.get_errors(y_data, label_data)

        # assertEqual() to check that the rmse values are equal to zero
        self.assertEqual(rmse.tolist(), np.zeros(np.shape(rmse)).tolist(), "perfect predictions should have zero rmse")

    def test_incorrect_prediction_errors_unittest(self):
        # test for when predictions and labels are notequal
        y_data = np.ones((100, 5))
        y_data[:, -1] = 0.
        label_data = 3. * np.ones((100, 2))
        rmse, rmse_cons = prediction_metrics.get_errors(y_data, label_data)

        # assertNotEqual() to check that the rmse values are not equal to zero
        self.assertNotEqual(rmse.tolist(), 0.0, "should get nonzero errors for wrong predictions")


def test_compute_nll():

    # test that the negative log likelihood is very small when the prediction has near zero uncertainty
    y_data = np.zeros((100, 5))
    y_data[:, 2:4] = .00001
    label_data = np.zeros((100, 2))
    nll_vec = prediction_metrics.compute_nll(y_data, label_data)

    assert np.all(nll_vec < 1.e-5)
