#!/usr/bin/python
# -*- coding: iso-8859-15 -*-


# --------------------------
# Bayesian Optimisation Code
# --------------------------

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from scipy.optimize import minimize
from pyDOE import *
from sklearn.model_selection import cross_val_score


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)




# -----------------------------------------
# --- Class for a continuous hyperparameter
# -----------------------------------------

# --- Define a hyperparameter class that contains all the required specs of the hyperparameter
class hyperparam(object):

    def __init__(self, list_in):

        # Initiate with 2 types of variable. We either specify bounds
        # for continuous variable or values for discrete. Note that for
        # now the values must be integers and be a list of consecutive
        #  integers.
        if len(list_in) == 2:
            self.bounds = list_in
            self.kind = 'continuous'
        elif len(list_in) > 2:
            self.bounds = [list_in[0], list_in[-1]]
            self.kind = 'discrete'


class iteration(object):

    def __init__(self, pars):

        #         # --- Sample data
        self.Xt = pars.Xt
        self.Yt = pars.Yt

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = self.propose_location(pars)
        # Convert to int where necessary

        # We need to recreate a dictionary with the keys given by the hyperparameter name before pasing into our
        # ML model
        self.X_nextdict = {}
        for i, hps1 in enumerate(sorted(pars.Xtdict.keys())):
            if pars.hps[hps1].kind == 'discrete':
                X_next[i] = int(X_next[i])
                self.X_nextdict[hps1] = X_next[i]
            else:
                self.X_nextdict[hps1] = X_next[i]

        # X_next = np.array(X_next,ndmin=(2)).reshape(1,-1)
        Y_next = pars.objF(self.X_nextdict)

        # Add the new sample point to the existing for the next iteration
        self.Xt = np.vstack((self.Xt, X_next.reshape(1, -1)[0]))
        self.Yt = np.concatenate((self.Yt, Y_next))

    # Sampling function to find the next values for the hyperparameters
    def propose_location(self, pars):

        # Proposes the next sampling point by optimizing the acquisition function. 
        # Args: acquisition: Acquisition function. 
        # X_sample: Sample locations (n x d). 
        # Y_sample: Sample values (n x 1). 
        # gpr: A GaussianProcessRegressor fitted to samples. 
        
        # Returns: Location of the acquisition function maximum. 
        
        self.N_hps = pars.Xt.shape[1]
        min_val = 1
        min_x = None

        self.gpr = pars.gpr
        self.Xt = pars.Xt

        # Find the best optimum by starting from n_restart different random points.
        Xs = lhs(self.N_hps, samples=pars.n_restarts, criterion='centermaximin')
        for i, hp in enumerate(sorted(pars.hps.keys())):
            Xs[:, i] = Xs[:, i] * (pars.hps[hp].bounds[1] - pars.hps[hp].bounds[0]) + pars.hps[hp].bounds[0]

            # Convert int values to integers
            if pars.hps[hp].kind == 'discrete':
                Xs[:, i] = Xs[:, i].astype(int)

        # Find the maximum in the acquisition function
        if pars.optim_rout == 'minimize':
            for x0 in Xs:
                res = minimize(self.min_obj, x0=x0, bounds=pars.bounds, method=pars.method) 
                # Find the best optimum across all initiations
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x   
                    
        elif pars.optim_rout == 'MCMC-MH':
            for x0 in Xs:
                res_x,res_f = self.MetroHastings(x0,[0.1]*self.N_hps,10000,tuple(pars.bounds))
                if res_f < min_val:
                    min_val = res_f
                    min_x = res_x

        return min_x.reshape(-1, 1)

    def min_obj(self, X):
        # Minimization objective is the negative acquisition function
        return -self.expected_improvement(X.reshape(-1, self.N_hps))

    # Acquisition function - here we use expected improvement
    def expected_improvement(self, X):

        # --- Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process 
        # surrogate model. 
        # X: Points at which EI shall be computed (m x d). 
        # X_sample: Sample locations (n x d). 
        # Y_sample: Sample values (n x 1). 
        # gpr: A GaussianProcessRegressor fitted to samples. 
        # xi: Exploitation-exploration trade-off parameter. 
        # .   - xi ~ O(0) => exploitation
        # .   - xi ~ O(1) => exploration
        # Returns: Expected improvements at points X.

        # Evaluate the Gaussian Process at a test location X to get the mean and std
        mu, sigma = self.gpr.predict(X, return_std=True)
        # Evaluate the Gaussian Process at the sampled points - this gets the mean values without the noise
        mu_sample = self.gpr.predict(self.Xt)

        sigma = sigma.reshape(-1, 1)  # self.Xt.shape[1])

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        mu_sample_opt = np.max(mu_sample)

        imp = mu - mu_sample_opt
        Z = imp / sigma

        Ei = (mu - mu_sample_opt) * norm.cdf(mu, loc=mu_sample_opt, scale=sigma) \
             + mu_sample_opt * norm.pdf(mu, loc=mu_sample_opt, scale=sigma)

        return Ei

    
    def MetroHastings(self,x0,sig,Niter,bounds):
    
        "Function to perform metropolis Hastings sampling in an MCMC"

        # --- Input ---
        # x0: initial guess for random walk - list of continuous variables
        # sig is the uncertainty in the MH sampling algorithm
        # Niter is number of iterations to perform
        # bounds: list of tuples of length x0, each one being the lower and upper bounds 

        # --- Output ---
        # Modal solution from the MCMC

        # Calculate initial guess
        acq = np.zeros(Niter)
        acq[0] = self.min_obj(x0.reshape(1,-1))

        # proposition point
        xp = np.zeros((len(x0),Niter))
        xp[:,0] = x0

        for iiter in range(1,Niter):

            # Propose new data point to try using MH
            for i in range(len(x0)):

                # iterate until we get a point in the correct interval
                if x0[i]<bounds[i][0]:
                    loc0 = bounds[i][0]
                elif x0[i]>bounds[i][1]:
                    loc0 = bounds[i][1]
                else:
                    loc0 = x0[i]

                Pnext = np.random.normal(loc=loc0,scale=sig[i])
                while (Pnext < bounds[i][0]) | (Pnext >= bounds[i][1]):
                    Pnext = np.random.normal(loc=loc0,scale=sig[i])

                # Then choose the first point that is    
                xp[i,iiter] = Pnext

            # Test value at this point
            acq[iiter] = self.min_obj(xp[:,iiter].reshape(1,-1))

            # Check if proposed point is better
            if acq[iiter] > acq[iiter-1]:
                x0 = xp[:,iiter].copy()

            else:
                p0 = [acq[iiter-1]/(acq[iiter]+acq[iiter-1]),acq[iiter]/(acq[iiter]+acq[iiter-1])]
                nextP = np.random.choice([0,1],p=p0)

                if nextP == 1:
                    x0 = xp[:,iiter].copy()
                else:
                    x0 = xp[:,iiter-1].copy()
                    
            
        # Now get optimal solution by fitting a histogram to the data - ignore first 10% of samples
        optim_x = np.zeros((1,len(x0)))   
        for i in range(optim_x.shape[1]):
            optim_x[0,i] = self.kernel_density_estimation(xp[i,int(0.1*Niter):],Niter)

        return optim_x,self.min_obj(optim_x.reshape(1,-1))

    def kernel_density_estimation(self,xpi,Niter):

        " Function to find peak in a kernel density "

        # We initially fudge this to get it working! 
        # So we fit a histogram and then find the middle of the tallest bar

        # Fit a histogram
        data = xpi.copy()
        data.sort()
        hist, bin_edges = np.histogram(data, density=True,bins=max(10,30))

        # Return the middle of the largest bin
        n = np.argmax(hist)
        return np.mean(bin_edges[n:n+2])
    
    
class BayesianOptimisation(object):

    def __init__(self, **kwargs):

        # Get hyperparameter info and convert to hyperparameter class
        self.hps = {}
        for hp in kwargs['hps'].keys():
            self.hps[hp] = hyperparam(kwargs['hps'][hp])

        # Objective function to minimise
        self.MLmodel = kwargs['MLmodel']

        # Number of hyperparameters
        N_hps = len(self.hps.keys())

        # --- Initial sample data
        if 'NpI' in kwargs.keys():
            self.NpI = kwargs['NpI']
        else:
            self.NpI = 2 ** N_hps
            
        # --- Optimisation routine for the acquisition function
        if 'optim_rout' in kwargs.keys():
            self.optim_rout = kwargs['optim_rout']
        else:
            self.optim_rout = 'minimize'
            
        # Get training data
        self.X_train = kwargs['X_train']
        self.y_train = kwargs['y_train']

        # Establish a dictionary for our hyperparameter values that we sample
        self.Xtdict = {}
        # ...and then an array for the same thing but with each column being
        # a different hyperparameter and ordered alphabetically
        self.Xt = np.zeros((self.NpI, len(self.hps.keys())))
        # We also need to collect together all of the bounds for the optimization routing into one array
        self.bounds = np.zeros((len(self.hps.keys()), 2))

        # Get some initial samples on the unit interval
        Xt = lhs(len(self.hps.keys()), samples=self.NpI, criterion='centermaximin')

        # For each hyper parameter, rescale the unit inverval on the 
        # appropriate range for that hp and store in a dict
        for i, hp in enumerate(sorted(self.hps.keys())):
            self.Xtdict[hp] = self.hps[hp].bounds[0] + Xt[:, i] * (self.hps[hp].bounds[1] - self.hps[hp].bounds[0])
            # convert these to an int if kind = 'discrete'
            
            if self.hps[hp].kind == 'discrete':
                self.Xtdict[hp] = self.Xtdict[hp].astype(int)

            self.bounds[i, :] = self.hps[hp].bounds

            self.Xt[:, i] = self.Xtdict[hp]

        # Calculate objective function at the sampled points
        self.Yt = self.objF(pars=self.Xtdict, n=self.NpI)

        # --- Number of iterations
        if 'Niter' in kwargs.keys():
            self.Niter = kwargs['Niter']
        else:
            self.Niter = 10 * N_hps
        logging.info('Will perform {} iterations'.format(self.Niter))

        # --- Number of optimisations of the acquisition function
        if 'n_restarts' in kwargs.keys():
            self.n_restarts = kwargs['n_restarts']
        else:
            self.n_restarts = 25 * N_hps

        # --- Optimisation method used
        if 'method' in kwargs.keys():
            self.method = kwargs['method']
        else:
            self.method = 'L-BFGS-B'

        # --- Define the Gaussian mixture model
        if 'kernel' in kwargs.keys():
            self.kernel = kwargs['kernel']
        else:
            self.kernel = RBF()

        if 'noise' in kwargs.keys():
            self.noise = kwargs['noise']
        else:
            self.noise = noise = 0.2

        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=noise ** 2)

    def optimise(self):
        for i in range(self.Niter):
            logging.info('Iteration {}'.format(i))
            it1 = iteration(self)
            self.Xt = it1.Xt
            self.Yt = it1.Yt
            print('current accuracy:', self.Yt[-1])
            print('best accuracy:', max(self.Yt))
            self.gpr.fit(self.Xt, self.Yt)

        # Print out best result
        max_val = max(self.Yt)
        best_params_vals = self.Xt[np.where(self.Yt==max_val)[0][0]]
        logging.info('Best result {}: Params: {}'.format(max_val, best_params_vals))
        best_params = {}
        for key, val in zip(self.MLmodel.get_params(), best_params_vals):
            best_params[key] = val
        logging.info('Best result {}: Params: {}'.format(max_val, best_params))
        return self

    def objF(self, pars, **kwargs):

        # Number of hyperparameter values to try.
        n = 1
        if 'n' in kwargs.keys():
            n = kwargs['n']

        # Initiate array to accumate the accuracy of the model
        sc = np.zeros(n)

        # Establish the basic ML model
        model = self.MLmodel


        for i in range(n):

            # Get dictionary of hyperparameter values to test at the ith iteration
            hps_iter = {}
            for hp in pars.keys():
                if self.hps[hp].kind == 'discrete':
                    hps_iter[hp] = int(pars[hp][i])
                else:
                    hps_iter[hp] = pars[hp][i]

            # Create instance of MLmodel with the hps at this iteration
            model.set_params(**hps_iter)

            # Train
            model.fit(self.X_train, self.y_train)

            # Score
            sc[i] = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=5))

        return sc