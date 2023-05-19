import time
from typing import List, Dict

import numpy as np
import pymc as mc
# import theano as th
# import theano.tensor as tt

import pytensor as th
import pytensor.tensor as tt

# import aesara
# import aesara.tensor as at
# from aesara.tensor import nnet # https://github.com/aesara-devs/aesara/issues/674 may deprecate soon (sad)
# TO DO: https://pytensor.readthedocs.io/en/latest/library/tensor/basic.html#pytensor.tensor.switch for relu
# from theano.tensor import nnet
# from multiprocessing import set_start_method

# https://pytensor.readthedocs.io/en/latest/library/config.html#libdoc-config
th.config.exception_verbosity='high'
th.config.optimizer="fast_run" # default
th.config.compute_test_value = "off" # "ignore"

class Sampler(object):
    def __init__(self, n_query:int, dim_features:int, update_func:str="pick_best", beta_demo:float=0.1, beta_pref:float=1.):
        """
        Initializes the sampler.

        :param n_query: Number of queries.
        :param dim_features: Dimension of feature vectors.
        :param update_func: options are "rank", "pick_best", and "approx". To use "approx", n_query must be 2. Will throw an assertion
            error otherwise.
        :param beta_demo: parameter measuring irrationality of human in providing demonstrations
        :param beta_pref: parameter measuring irrationality of human in selecting preferences
        """
        self.n_query = n_query
        self.dim_features = dim_features
        self.update_func = update_func
        self.beta_demo = beta_demo
        self.beta_pref = beta_pref

        if self.update_func=="approx":
            assert self.n_query == 2, "Cannot use approximation to update function if n_query > 2"
        elif not (self.update_func=="rank" or self.update_func=="pick_best"):
            raise Exception(update_func + " is not a valid update function.")

        # feature vectors from demonstrated trajectories
        self.phi_demos = np.zeros((1, self.dim_features))
        # a list of np.arrays containing feature difference vectors and which encode the ranking from the preference
        # queries
        self.phi_prefs = []

        self.f = None

    def load_demo(self, phi_demos:np.ndarray):
        """
        Loads the demonstrations into the Sampler.

        :param demos: a Numpy array containing feature vectors for each demonstration.
            Has dimension n_dem x self.dim_features.
        """
        self.phi_demos = phi_demos

    def load_prefs(self, phi: Dict, rank):
        """
        Loads the results of a preference query into the sampler.

        :param phi: a dictionary mapping rankings (0,...,n_query-1) to feature vectors.
        """
        result = []
        if self.update_func == "rank":
            result = [None] * len(rank)
            for i in range(len(rank)):
                result[i] = phi[rank[i]]
        elif self.update_func == "approx":
            result = phi[rank] - phi[1-rank]
        elif self.update_func == "pick_best":
            result, tmp = [phi[rank] - phi[rank]], []
            for key in sorted(phi.keys()):
                if key != rank:
                    tmp.append(phi[key] - phi[rank])
            result.extend(tmp)
        self.phi_prefs.append(np.array(result))


    def clear_pref(self):
        """
        Clears all preference information from the sampler.
        """
        self.phi_prefs = []

    def sample(self, N:int, T:int=1, burn:int=1000) -> np.array:
        """
        Returns N samples from the distribution defined by applying update_func on the demonstrations and preferences
        observed thus far.

        :param N: number of samples to draw.
        :param T: if greater than 1, all samples except each T^{th} sample are discarded. --> thin
        :param burn: how many samples before the chain converges; these initial samples are discarded. --> tune

        :return: list of samples drawn.
        """


        # ----- OLD - pre-compile sampling function ----- #
        # As per https://discourse.pymc.io/t/expected-an-array-like-object-but-found-a-variable/8445/3 and 
        # https://discourse.pymc.io/t/aesara-error-expected-an-array-like-object-but-found-a-variable/10910 and
        # https://discourse.pymc.io/t/help-creating-custom-op/10966/3
        # There is no need to compile beforehand. "PyMC will then compile the right functions itself when it needs to. 
        # As long as any operation in [function] uses aesara operators (we overload most of the Python operators, so 
        # this also applies to +, * and so on) you should be fine."
        # th.config.compute_test_value = 'warn'
        # th.config.print_test_value = True

        # x = tt.vector("x")
        # # x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)
        # # # print(type(x))
        # # # print(x.__repr__())
        # # # print("OK")

        # # # define update function
        # start = time.time()
        # if self.update_func=="approx":
        #     self.f = th.function([x], tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
        #                     + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #     # self.f = th.function(inputs=[x], outputs=tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
        #     #                 + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)),
        #     #                 mode="DebugMode")
        #     # print("Approx f fine")
        #     # print(tt.sum([-at.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))]))
        #     # print(tt.sum([-at.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
        #     #                 + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #     # print(self.f(x.tag.test_value))

        # elif self.update_func=="pick_best":
        #     self.f = th.function([x], tt.sum(
        #         [-tt.log(tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i], x)))) for i in range(len(self.phi_prefs))])
        #                     + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #     # print("Pick best f fine")
        # elif self.update_func=="rank":
        #     self.f = th.function([x], tt.sum( # summing across different queries
        #         [tt.sum( # summing across different terms in PL-update
        #             -tt.log(
        #                 [tt.sum( # summing down different feature-differences in a single term in PL-update
        #                     tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i][j:, :] - self.phi_prefs[i][j], x))
        #                 ) for j in range(self.n_query)]
        #             )
        #         ) for i in range(len(self.phi_prefs))])
        #                     + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #     # print("Rank f fine")
        # print("Finished constructing sampling function in " + str(time.time() - start) + "seconds")

        # ----- DEBUG ATTEMPT: Define custom distribution defined by applying update_func on demos and prefs ----- #
        # def dist_f(x: tt.TensorVariable) -> th.compile.function:
        #     if self.update_func=="approx":
        #         # self.f = th.function([x], tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
        #         #                 + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #         self.f = th.function([x], tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))])
        #                         + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))

        #     elif self.update_func=="pick_best":
        #         self.f = th.function([x], tt.sum(
        #             [-tt.log(tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i], x)))) for i in range(len(self.phi_prefs))])
        #                         + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #         print("Pick best f fine")
        #     elif self.update_func=="rank":
        #         self.f = th.function([x], tt.sum( # summing across different queries
        #             [tt.sum( # summing across different terms in PL-update
        #                 -tt.log(
        #                     [tt.sum( # summing down different feature-differences in a single term in PL-update
        #                         tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i][j:, :] - self.phi_prefs[i][j], x))
        #                     ) for j in range(self.n_query)]
        #                 )
        #             ) for i in range(len(self.phi_prefs))])
        #                         + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x)))
        #     return self.f
        # perform sampling
        # print("HARDY", -np.ones(self.dim_features))
        # lower = tt.as_tensor_variable(-np.ones(self.dim_features))
        # upper = tt.as_tensor_variable(np.ones(self.dim_features))
        # value = tt.as_tensor_variable(np.zeros(self.dim_features))
        # print(upper)
        # print(lower)
        # shape = self.dim_features
        # print(f"Running on PyMC v{mc.__version__}")

        # print("Defined distribution outside of context manager")
        # x = mc.Uniform.dist(name='x', lower=-np.ones(self.dim_features), upper=np.ones(self.dim_features))
        # x.tag.test_value = np.zeros(self.dim_features)

        # print("Try doing function on this distribution")
        # print(self.f(x.tag.test_value))
        # print("Worked!")

        # ---- Define relu since they are deprecating it ----- #
        def relu(x, alpha=0):
            """
            Compute the element-wise rectified linear activation function.

            .. versionadded:: 0.7.1

            Parameters
            ----------
            x : symbolic tensor
                Tensor to compute the activation function for.
            alpha : `scalar or tensor, optional`
                Slope for negative input, usually between 0 and 1. The default value
                of 0 will lead to the standard rectifier, 1 will lead to
                a linear activation function, and any value in between will give a
                leaky rectifier. A shared variable (broadcastable against `x`) will
                result in a parameterized rectifier with learnable slope(s).

            Returns
            -------
            symbolic tensor
                Element-wise rectifier applied to `x`.

            Notes
            -----
            This is numerically equivalent to ``switch(x > 0, x, alpha * x)``
            (or ``maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
            formulation or an optimized Op, so we encourage to use this function.

            """
            # This is probably the fastest implementation for GPUs. Both the forward
            # pass and the gradient get compiled into a single GpuElemwise call.
            # TODO: Check if it's optimal for CPU as well; add an "if" clause if not.
            # TODO: Check if there's a faster way for the gradient; create an Op if so.
            if alpha == 0:
                return 0.5 * (x + abs(x))
            else:
                # We can't use 0.5 and 1 for one and half.  as if alpha is a
                # numpy dtype, they will be considered as float64, so would
                # cause upcast to float64.
                alpha = at.as_tensor_variable(alpha)
                f1 = 0.5 * (1 + alpha)
                f2 = 0.5 * (1 - alpha)
                return f1 * x + f2 * abs(x)


        # ----- Define log_P that was given in the DemPref paper ----- #
        
        # def logp(value, x):
        def logp(x: tt.TensorVariable):
            start = time.time()
            if self.update_func=="approx":
                # was -nnet.relu
                self.f = tt.sum([-relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))
            
            elif self.update_func=="pick_best":
                self.f = tt.sum([-tt.log(tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i], x)))) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))

            elif self.update_func=="rank":
                self.f = tt.sum([tt.sum(-tt.log([tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i][j:, :] - self.phi_prefs[i][j], x))) for j in range(self.n_query)])) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))
            
            print("Finished constructing sampling function in " + str(time.time() - start) + "seconds")
            return self.f

            # return self.f(value)
            # if (x**2).sum()>=1.:
            #     return -np.inf
            # else:
            #     return self.f(x)

        # def random(x, rng = None, size=None): # For custom distribution approach - didn't work
        #     return rng.uniform(low=-np.ones(self.dim_features), high=np.ones(self.dim_features), size=size)

        with mc.Model() as model:

            # ----- Define our prior: "uniform on the unit ball. i.e., w ∼ Uni f(B(0,1))" ----- #

            # OLD --> PyMC2 or 3 ish? No longer use "value" or "testvalue" parameters --> renamed to initval, see link below
            # https://www.pymc.io/projects/docs/en/v3/api/distributions/continuous.html#pymc3.distributions.continuous.Uniform
            # x = mc.Uniform(name='x', lower=-np.ones(self.dim_features), upper=np.ones(self.dim_features), value=np.zeros(self.dim_features))
            # In new versions of PyMC, we now define variables in a model context using with "mc.Model():"
        # x = mc.Uniform(name='x', lower=-np.ones(self.dim_features), upper=np.ones(self.dim_features), value=np.zeros(self.dim_features))

            x = mc.Uniform(name='x', lower=-np.ones(self.dim_features), upper=np.ones(self.dim_features), initval=np.zeros(self.dim_features))
            print("Defined x")
            # print(type(x)) # -> TensorVariable
            # th.dprint(x)

            # Draw some random variables - debug use only
            # for i in range(10):
            #     print(f"Sample {i}: {mc.draw(x)}")

            # "No info over w before observing any demonstrations"
            # Use potential as a hacky way to get around the definition of a likelihood
            # Defining a true likelihood would require inclusion of observations, which we don't have
            # All the samples that we draw are based on how we defined the potential
            # Can tack on a custom logp using Potential (see original post): https://discourse.pymc.io/t/custom-distribution-with-pm-customdist/11071
            p = mc.Potential("sphere", mc.math.switch((x**2).sum() >= 1., -np.inf, logp(x)))
            # p = mc.Potential("sphere", mc.math.switch((x**2).sum() >= 1., -np.inf, self.f(x)))
            print("Defined potential")

            # print((x**2).sum())

            # https://stackoverflow.com/questions/21876836/how-to-apply-a-custom-function-to-a-variable-in-pymc
            # https://discourse.pymc.io/t/drawing-values-from-custom-distribution/10395/5
            # TRY: @pm.deterministic
            # def sphere(x):
            #     if (x**2).sum() >= 1.:
            #         print("Hallo")
            #         return -np.inf
            #     else:
            #         return self.f(x)
            # mc.CustomDist("update_func", x, logp=logp, random=random, size=self.dim_features)
            # print("BOING")

            # start = time.time()
            # if self.update_func=="approx":
            #     self.f = tt.sum([-tt.nnet.relu(-self.beta_pref * tt.dot(self.phi_prefs[i], x)) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))
            
            # elif self.update_func=="pick_best":
            #     self.f = tt.sum([-tt.log(tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i], x)))) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))

            # elif self.update_func=="rank":                
            #     self.f = tt.sum([tt.sum(-tt.log([tt.sum(tt.exp(self.beta_pref * tt.dot(self.phi_prefs[i][j:, :] - self.phi_prefs[i][j], x))) for j in range(self.n_query)])) for i in range(len(self.phi_prefs))]) + tt.sum(self.beta_demo * tt.dot(self.phi_demos, x))
            
            # print("Finished constructing sampling function in " + str(time.time() - start) + "seconds")

            # constraint = (x**2).sum() >= 1.
            # constraint = mc.math.ge(mc.math.sum(mc.math.sqr(x)), 1.) # Equivalent to constraint = (x**2).sum() >= 1.
            # print(type(constraint))
            # print(self.f)
            # print(x.type)
            # print(x.__repr__())

            # print(model.initial_point()["x_interval__"]) # New way of accessing testvalue in PYMC (https://discourse.pymc.io/t/how-can-i-get-test-value-in-pymc-pymc4/10270/7)
            # print("Boop")
            # print(self.f(model.initial_point()["x_interval__"]))
            # print("Test")
            # p = mc.Potential(name="sphere", var=mc.math.switch((x**2).sum() >= 1., -np.inf, self.f(model.initial_point()["x_interval__"]))) # problem with self.f(x)
            # p = mc.Potential(name="sphere", var=sphere)
            # p = mc.Potential(name="sphere", var=mc.math.switch(constraint, -np.inf, dist_f(x)))
            # p = mc.Potential(name="sphere", var=mc.math.switch(constraint, -np.inf, self.f))

        # p = mc.Potential(
        #     logp = sphere,
        #     name = 'sphere',
        #     parents = {'x': x},
        #     doc = 'Sphere potential',
        #     verbose = 0)
            
            # print("Yoinks")

            # https://pymcmc.readthedocs.io/en/latest/modelfitting.html#markov-chain-monte-carlo-the-mcmc-class
            # This will create a MCMC model with x as component
            # The MCMC class implements PyMC’s core business: producing ‘traces’ for a model’s variables which, with careful thinning, can be considered independent joint samples from the posterior.
            # chain = mc.MCMC([x])

            # chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.dim_features)/5000)
            # chain.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
            # samples = x.trace()
            # samples = np.array([x/np.linalg.norm(x) for x in samples])


            # Look at https://towardsdatascience.com/bayesian-logistic-regression-with-pymc3-8e17c576f31a
            # AND https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html
            # step = mc.DEMetropolis(vars=[x], S=np.eye(self.dim_features)/5000)

            step = mc.DEMetropolisZ()
            print("Step definition okay")

            print("name: ", __name__)
            # __name__="__main__"
            trace = mc.sample(draws=N*T, tune=burn, step=step, discard_tuned_samples=True, return_inferencedata=False)
            print("Trace OK")
            # print(samples.sample_stats) # Works only if you set return_inferencedata=True (for return arviz.InferenceData trace)
            # print("Sample stats okay")
            samples = np.array([x/np.linalg.norm(x) for x in trace.get_values(varname="x", thin=T)])
            print("OKAY")

        # print("Finished MCMC after drawing " + str(N*T+burn) + " samples")
        print("Done sampling")
        return samples



# # # Comment section after debugging
if __name__ == "__main__":
    print('Main here')
    sampler = Sampler(n_query=2, dim_features=4, update_func="approx",beta_demo=0.1, beta_pref=5)
    s = sampler.sample(50000)

    print(type(s))


