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
DEFAULT_STEPS = 10

def init(options):
    return ParallelLinearDescentChooser(options)

class ParallelLinearDescentChooser(object):
    """"
    This optimization strategy of this chooser sets the same value for all parameters to the same value
    (hence the name parallel) and increases the value of the parameters in each step. The number of steps
    can be configured in the by setting the the property 'steps' in the 'chooser-args':

    "chooser-args": {
        "steps": 10
    },
    """

    def __init__(self, options):
        self.task_group  = None
        self.isFit = False
        self.nextTry = None
        self.num_dims = None
        if 'chooser-args' in options:
            steps = int(options['chooser-args'].get('steps', DEFAULT_STEPS))
        else:
            steps = DEFAULT_STEPS
        self.stepDecrease = 1.0/steps

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
                lastTry = 0.0
                self.nextTry = 0.0  # we try the minimum value first
            else:
                lastTry = hypers[task_name]['lastTry']
                self.nextTry = lastTry + self.stepDecrease

            # save hyper-parameters for next round
            new_hypers[task_name] = {'lastTry': self.nextTry}

        if VERBOSE:
            if self.nextTry is None:
                print "average improvement over the last three runs is negative."
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