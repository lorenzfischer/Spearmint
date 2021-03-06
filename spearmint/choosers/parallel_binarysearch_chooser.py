# -*- coding: utf-8 -*-

import sys
import numpy          as np
import numpy.random   as npr
import scipy.optimize as spo
import multiprocessing
import time

from collections import defaultdict

from .acquisition_functions  import compute_ei
from ..utils.grad_check      import check_grad
from ..grids                 import sobol_grid
from ..models.abstract_model import function_over_hypers
from ..                      import models

VERBOSE = False

def init(options):
    return ParallelBinarySarchChooser(options)


class ParallelBinarySarchChooser(object):
    """"The optimization strategy of this chooser is a parallel binary search, meaning that all parameters will be increased
        or decreased in parallel, following a binary search approach.
    """

    def __init__(self, options):
        self.task_group  = None
        self.isFit = False
        self.nextTry = None
        self.num_dims = None

    def fit(self, task_group, hypers=None, options=None):
        self.task_group = task_group
        self.num_dims   = task_group.num_dims
        new_hypers      = {}
        hypers          = hypers if hypers is not None else defaultdict(dict)

        # print 'Fittings tasks: %s' % str(task_group.tasks.keys())

        if len(task_group.tasks) > 1:
            raise RuntimeError("We can only handle one task for now")

        for task_name, task in task_group.tasks.iteritems():
            if len(task.values) == 0 or task_name not in hypers:
                rangeMin = 0.0
                rangeMax = 1.0
                self.nextTry = 0.0  # we try the minimum value first
            else:
                rangeMin = hypers[task_name]['min']
                rangeMax = hypers[task_name]['max']

                if len(task.values) == 1:  # we only have one data point (min run)
                    # we try the middle next and leave min and max as they are
                    self.nextTry = rangeMin + ((rangeMax - rangeMin) / 2)
                else:  # if we have more than one data point:
                    # compare the performance measurements:
                    # remember, the bayesian optimizer (default_optimizer) wants to minimize things, so we are looking
                    # for small values here as well. in other words: the smaller the performance value, the better.
                    oldMax = min(task.values[:-1])
                    performanceLast = task.values[-1]

                    # if it decreased (better performance), we look in the upper half -> set min to old mid
                    # if it increased (worse performance), we look in the lower half -> set max to old mid
                    old_middle = rangeMin + ((rangeMax - rangeMin) / 2)
                    if performanceLast <= oldMax:  # better performance
                        rangeMin = old_middle
                    elif performanceLast > oldMax: # worse performance
                        rangeMax = old_middle

                    self.nextTry = rangeMin + ((rangeMax - rangeMin) / 2)

            # save hyper-parameters for next round
            new_hypers[task_name] = {'min': rangeMin, 'max': rangeMax}

            # stopping condition: if min and max are (almost) the same, we stop the evaluation.
            if (rangeMax - rangeMin) < 0.001:  # good for ranges up to 10k
                self.nextTry = None

        if VERBOSE:
            if self.nextTry is None:
                print "max and min are too close, we don't try anything anymore."
            else:
                print 'Next try using %f' % self.nextTry

        time.sleep(5)  # delays for 5 seconds to give the currently running topo time to shut down

        self.isFit = True
        return new_hypers

    def suggest(self):
        """
        This method needs to return the parameters as they are suggested by this optimiser. Example:

        ndarray
        """
        sys.stderr.write('Getting suggestion...\n')

        if not self.isFit:
            raise Exception("You must call fit() before calling suggest()")

        # stopping condition, there is nothing to try here anymore
        if self.nextTry is None:
            return None

        # we use the from_unit method provided by the base_task class which computes
        # the suggestion for us. We only need to provide the method with a ndimx1 ndarray which contains values
        # between 0 and 1 representing which value in the range should be chosen next for each parameter (0.0 means the
        # minimum value, 0.5 means the value between min and max, and 1.0 means the maximum value for the parameter).
        unit_array = np.array([self.nextTry] * self.num_dims)

        suggestion = self.task_group.from_unit(unit_array)
        sys.stderr.write("\nSuggestion:     ")
        self.task_group.paramify_and_print(suggestion.flatten(), left_indent=16)

        return suggestion