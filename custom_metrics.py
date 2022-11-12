"""Define custom metrics.

Classes
-------
CustomMAE(tf.keras.metrics.Metric)
    Compute the mean absolute error.

InterquartileCapture(tf.keras.metrics.Metric)
    Compute the fraction of true values between the 25 and 75 percentiles.

class SignTest(tf.keras.metrics.Metric)
    Compute the fraction of true values above the median.

Usage
-----
* For example, include CustomMAE() as a metric in model.compile; e.g.

    model.compile(
        ...
        metrics=[CustomMAE()],
    )

Notes
-----
* The computations are done by maintaining running sums of total predictions
    and correct predictions made across all batches in an epoch. The
    running sums are reset at the end of each epoch.

"""
import silence_tensorflow.auto
import tensorflow as tf

import mahalanobis

__author__ = "Randal J Barnes and Elizabeth A. Barnes"
__version__ = "12 November 2022"


class CustomMAE(tf.keras.metrics.Metric):
    """Compute the prediction mean Euclidian error."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error = self.add_weight("error", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        error = tf.experimental.numpy.hypot(
            pred[:, 0] - y_true[:, 0],
            pred[:, 1] - y_true[:, 1],
        )

        batch_error = tf.reduce_sum(error)
        batch_total = len(y_true[:, 0])

        self.error.assign_add(tf.cast(batch_error, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.error / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class InterquartileCapture(tf.keras.metrics.Metric):
    """Compute the fraction of true values between the 25 and 75 percentiles.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        batch_cdf = mahalanobis.compute_cdf(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            pred[:, 3],
            pred[:, 4],
            y_true[:, 0],
            y_true[:, 1],
        )

        batch_count = tf.reduce_sum(
            tf.cast(
                tf.math.logical_and(
                    tf.math.greater(batch_cdf, 0.25),
                    tf.math.less(batch_cdf, 0.75)
                ),
                tf.float32
            )

        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class SignTest(tf.keras.metrics.Metric):
    """Compute the fraction of true values outside of the 0.50
    Mahalanobis ellipse.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight("count", initializer="zeros")
        self.total = self.add_weight("total", initializer="zeros")

    def update_state(self, y_true, pred, sample_weight=None):
        cdf = mahalanobis.compute_cdf(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            pred[:, 3],
            pred[:, 4],
            y_true[:, 0],
            y_true[:, 1],
        )
        outside = (cdf > 0.50)

        batch_count = tf.reduce_sum(
            tf.cast(outside, tf.float32)
        )
        batch_total = len(y_true[:, 0])

        self.count.assign_add(tf.cast(batch_count, tf.float32))
        self.total.assign_add(tf.cast(batch_total, tf.float32))

    def result(self):
        return self.count / self.total

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
