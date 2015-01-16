# -*- coding: utf-8 -*-

import sys
import numpy          as np
import numpy.random   as npr
import scipy.optimize as spo
import multiprocessing

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
                min = 0.0
                max = 1.0
                self.nextTry = 0.0  # we try the minimum value first
            else:
                min = hypers[task_name]['min']
                max = hypers[task_name]['max']

                if len(task.values) == 1:  # we only have one data point (min run)
                    # we try the middle next and leave min and max as they are
                    self.nextTry = min + ((max - min) / 2)
                else:  # if we have more than one data point:
                    # compare the performance measurements:
                    # remember, the bayesian optimizer (default_optimizer) wants to minimize things, so we are looking
                    # for small values here as well. in other words: the smaller the performance value, the better.
                    performanceOld = task.values[-2]
                    performanceLast = task.values[-1]

                    # if it decreased (better performance), we look in the upper half -> set min to old mid
                    # if it increased (worse performance), we look in the lower half -> set max to old mid
                    old_middle = min + ((max - min) / 2)
                    if performanceLast <= performanceOld:  # better performance
                        min = old_middle
                    elif performanceLast > performanceOld: # worse performance
                        max = old_middle

                    self.nextTry = min + ((max - min) / 2)

            # save hyper-parameters for next round
            new_hypers[task_name] = {'min': min, 'max': max}

            # stopping condition: if min and max are (almost) the same, we stop the evaluation.
            if (max - min) < 0.001:  # good for ranges up to 10k
                self.nextTry = None

        if VERBOSE:
            if self.nextTry is None:
                print "max and min are too close, we don't try anything anymore."
            else:
                print 'Next try using %f' % self.nextTry

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

        # here we do the binary search, we use the from_unit method provided by the base_task class which computes
        # the suggestion for us. We only need to provide the method with a ndimx1 ndarray which contains values
        # between 0 and 1 representing which value in the range should be chosen next for each parameter (0.0 means the
        # minimum value, 0.5 means the value between min and max, and 1.0 means the maximum value for the parameter).
        #
        # Hence, we supply the ndarray based on the previous run, by either
        #
        #
        #
        unit_array = np.array([self.nextTry] * self.num_dims)

        suggestion = self.task_group.from_unit(unit_array)
        sys.stderr.write("\nSuggestion:     ")
        self.task_group.paramify_and_print(suggestion.flatten(), left_indent=16)

        return suggestion


    # TODO: add optimization in here
    def best(self):
        grid = self.grid
        obj_task = self.task_group.tasks[self.objective['name']]
        obj_model = self.models[self.objective['name']]

        # If unconstrained
        if self.numConstraints() == 0:
            # Compute the GP mean
            obj_mean, obj_var = obj_model.function_over_hypers(obj_model.predict, grid)

            # find the min and argmin of the GP mean
            current_best_location = grid[np.argmin(obj_mean),:][None]
            best_ind = np.argmin(obj_mean)
            current_best_value = obj_mean[best_ind]
            std_at_best = np.sqrt(obj_var[best_ind])

            # un-normalize the min of mean to original units
            unnormalized_best_value = obj_task.unstandardize_mean(obj_task.unstandardize_variance(current_best_value))
            unnormalized_std_at_best = obj_task.unstandardize_variance(std_at_best)

            # Print out the minimum according to the model
            sys.stderr.write('\nMinimum expected objective value under model is %.5f (+/- %.5f), at location:\n' % (unnormalized_best_value, unnormalized_std_at_best))
            self.task_group.paramify_and_print(self.task_group.from_unit(current_best_location).flatten(), left_indent=16, indent_top_row=True)

            # Compute the best value seen so far
            vals = self.task_group.values[self.objective['name']]
            inps = self.task_group.inputs
            best_observed_value = np.min(vals)
            best_observed_location = inps[np.argmin(vals),:][None]

            # Don't need to un-normalize inputs here because these are the raw inputs
            sys.stderr.write('\nMinimum of observed values is %f, at location:\n' % best_observed_value)
            self.task_group.paramify_and_print(best_observed_location.flatten(), left_indent=16, indent_top_row=True)

        else:

            mc = self.probabilistic_constraint(grid)
            if not np.any(mc):
                # P-con is violated everywhere
                # Compute the product of the probabilities, and return None for the current best value
                probs = reduce(lambda x,y:x*y, [self.confidence(c, grid) for c in self.constraints], np.ones(grid.shape[0]))
                best_probs_ind = np.argmax(probs)
                best_probs_location = grid[best_probs_ind,:][None]
                # TODO -- could use BFGS for this (unconstrained) optimization as well -- everytime for min of mean

                sys.stderr.write('\nNo feasible region found (yet).\n')
                sys.stderr.write('Maximum probability of satisfying constraints = %f\n' % np.max(probs))
                sys.stderr.write('At location:    ')
                self.task_group.paramify_and_print(self.task_group.from_unit(best_probs_location).flatten(), left_indent=16)

                return None, best_probs_location

            # A feasible region has been found

            # Compute GP mean and find minimum
            mean, var = obj_model.function_over_hypers(obj_model.predict, grid)
            valid_mean = mean[mc]
            valid_var = var[mc]
            best_ind = np.argmin(valid_mean)
            current_best_location = (grid[mc])[best_ind,:][None]
            ind = np.argmin(valid_mean)
            current_best_value = valid_mean[ind]
            std_at_best = np.sqrt(valid_var[ind])

            unnormalized_best = obj_task.unstandardize_mean(obj_task.unstandardize_variance(current_best_value))
            unnormalized_std_at_best = obj_task.unstandardize_variance(std_at_best) # not used -- not quite
            # right to report this -- i mean there is uncertainty in the constraints too
            # this is the variance at that location, not the standard deviation of the minimum...
            # not sure if this distinction is a big deal

            sys.stderr.write('\nMinimum expected objective value satisfying constraints w/ high prob: %f\n' % unnormalized_best)
            sys.stderr.write('At location:    ')
            self.task_group.paramify_and_print(self.task_group.from_unit(current_best_location).flatten(), left_indent=16)

            # Compute the best value seen so far
            with np.errstate(invalid='ignore'):
                all_constraints_satisfied = np.all(np.greater(np.array([x.values for x in self.task_group.tasks.values()]), 0), axis=0)
            if not np.any(all_constraints_satisfied):
                sys.stderr.write('No observed result satisfied all constraints.\n')
            else:
                inps = self.task_group.inputs
                vals = self.task_group.values[self.objective['name']]
                # get rid of those that violate constraints
                vals[np.logical_not(all_constraints_satisfied)] = np.max(vals)
                # get rid of NaNs -- set them to biggest not-nan value, then they won't be the minimum
                vals[np.isnan(vals)] = np.max(vals[np.logical_not(np.isnan(vals))])
                best_observed_value = np.min(vals)
                best_observed_location = inps[np.argmin(vals),:][None]
                # Don't need to un-normalize inputs here because these are the raw inputs
                sys.stderr.write('\nBest observed values satisfying constraints is %f, at location:\n' % best_observed_value)
                self.task_group.paramify_and_print(best_observed_location.flatten(), left_indent=16, indent_top_row=True)


        # Return according to model, not observed
        return current_best_value, current_best_location

    def numConstraints(self):
        return len(self.constraints)

    # The confidence that conststraint c is satisfied
    def confidence(self, c, grid, compute_grad=False):
        return self.models[c].function_over_hypers(self.models[c].pi, grid, compute_grad=compute_grad)

    # Returns a boolean array of size pred.shape[0] indicating whether the prob con-constraint is satisfied there
    def probabilistic_constraint(self, pred):
        return reduce(np.logical_and,
            [self.confidence(c, pred) >= self.task_group.tasks[c].options.get('min-confidence', 0.99)
                for c in self.constraints],
                np.ones(pred.shape[0], dtype=bool))

    def acquisition_function_over_hypers(self, *args, **kwargs):
        return function_over_hypers(self.models.values(), self.acquisition_function, *args, **kwargs)

    def acquisition_function(self, cand, current_best, compute_grad=True):
        obj_model = self.models[self.objective['name']]

        # If unconstrained, just compute regular ei
        if self.numConstraints() == 0:
            return compute_ei(obj_model, cand, ei_target=current_best, compute_grad=compute_grad)

        if cand.ndim == 1:
            cand = cand[None]

        N_cand = cand.shape[0]

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############   Part that depends on the objective     ############
        ##############                                          ############
        ############## ---------------------------------------- ############
        if current_best is None:
            ei = 1.
            ei_grad = 0.
        else:
            target = current_best

            # Compute the predictive mean and variance
            if not compute_grad:
                ei = compute_ei(obj_model, cand, target, compute_grad=compute_grad)
            else:
                ei, ei_grad = compute_ei(obj_model, cand, target, compute_grad=compute_grad)

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############  Part that depends on the constraints    ############
        ##############                                          ############
        ############## ---------------------------------------- ############
        # Compute p(valid) for ALL constraints
        p_valid, p_grad = list(), list()
        for c in self.constraints:
            if compute_grad:
                pv, pvg = self.models[c].pi(cand, compute_grad=True)
                p_valid.append(pv)
                p_grad.append(pvg)
            else:
                p_valid.append(self.models[c].pi(cand, compute_grad=False))

        p_valid_prod = reduce(np.multiply, p_valid, np.ones(N_cand))

        # To compute the gradient, need to do the chain rule for the product of N factors
        if compute_grad:
            p_grad_prod = np.zeros(p_grad[0].shape)
            for i in xrange(self.numConstraints()):
                pg = p_grad[i]
                for j in xrange(self.numConstraints()):
                    if j == i:
                        continue
                    pg *= p_valid[j]
                p_grad_prod += pg
            # multiply that gradient by all other pv's (this might be numerically disasterous if pv=0...)

        ############## ---------------------------------------- ############
        ##############                                          ############
        ##############    Combine the two parts (obj and con)   ############
        ##############                                          ############
        ############## ---------------------------------------- ############

        acq = ei * p_valid_prod

        if not compute_grad:
            return acq
        else:
            return acq, ei_grad * p_valid_prod + p_grad_prod * ei

    # Flip the sign so that we are maximizing with BFGS instead of minimizing
    def acq_optimize_wrapper(self, cand, current_best, compute_grad):
        ret = self.acquisition_function_over_hypers(cand, current_best, compute_grad=compute_grad)

        if isinstance(ret, tuple) or isinstance(ret, list):
            return (-ret[0],-ret[1].flatten())
        else:
            return -ret

    def optimize_pt(self, initializer, bounds, current_best, compute_grad=True):
        opt_x, opt_y, opt_info = spo.fmin_l_bfgs_b(self.acq_optimize_wrapper,
                initializer.flatten(), args=(current_best,compute_grad),
                bounds=bounds, disp=0, approx_grad=(not compute_grad))
        return opt_x