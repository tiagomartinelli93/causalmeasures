from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional, KDEMultivariate, EstimatorSettings
import pandas as pd
from itertools import product
from scipy.integrate import nquad
from scipy import stats
import numpy as np

class CausalEffect(object):
    def __init__(self, X, causes, effects, confounders=[], variable_types=None, info=None):
        """
        We want to calculate the causal effect of X on Y through
        back-door adjustment, P(Y|do(X)) = Sum( P(Y|X,Z)P(Z), Z) 
        for some admissable set of control variables, Z.  First we 
        calculate the conditional density P(Y|X,Z), then the density
        P(Z).  We find the support of Z so we can properly sum over
        it later. The variable_types are a dictionary with the column name
        pointing to an element of set(['o', 'u', 'c']), for 'ordered',
        'unordered discrete', or 'continuous'. info=True gives the causal effect
        from an informational perspective see local_flow function for detals
        """
        conditionals = confounders + causes 
        self.causes = causes
        self.effects = effects
        self.confounders = confounders
        self.conditionals = conditionals

        if len(X) > 300 or max(len(causes+confounders),len(effects+confounders)) >= 3:
            self.defaults=EstimatorSettings(n_jobs=4, efficient=True)
        else:
            self.defaults=EstimatorSettings(n_jobs=-1, efficient=False)
        
        if variable_types:
            self.variable_types = variable_types
            dep_type      = [variable_types[var] for var in effects]
            indep_type    = [variable_types[var] for var in conditionals]

        else:
            self.variable_types = self.__infer_variable_types(X)

        if 'c' not in variable_types.values():
            bw = 'cv_ml'
        
        else:
            bw = 'normal_reference'

        
        self.kYgXZ = KDEMultivariateConditional(endog=X[effects],
                                                         exog=X[conditionals],
                                                         dep_type=''.join(dep_type),
                                                         indep_type=''.join(indep_type),
                                                         bw=bw,
                                                         defaults=EstimatorSettings(n_jobs=4))

        self.support = self.get_support(X)
        
        self.discrete_variables = [variable for variable, var_type in self.variable_types.items() if var_type in ['o', 'u']]
        self.continuous_variables = [variable for variable, var_type in self.variable_types.items() if var_type == 'c']     
            
        if confounders:       
            Z_types = [variable_types[var] for var in confounders]
            self.discrete_Z = list(set(self.discrete_variables).intersection(set(confounders)))
            self.continuous_Z = list(set(self.continuous_variables).intersection(set(confounders))) 
            
            self.kZ = KDEMultivariate(X[confounders], 
                                  var_type=''.join(Z_types),
                                  bw=bw,
                                  defaults=EstimatorSettings(n_jobs=4))
            
        if info:        
            X_types = [variable_types[var] for var in causes]
            self.discrete_X = list(set(self.discrete_variables).intersection(set(causes)))
            self.continuous_X = list(set(self.continuous_variables).intersection(set(causes)))
        
            self.kX = KDEMultivariate(X[causes], 
                                  var_type=''.join(X_types),
                                  bw=bw,
                                  defaults=EstimatorSettings(n_jobs=4))

 
    def get_support(self, X):
        """
        find the smallest cube around which the densities are supported,
        allowing a little flexibility for variables with small/large bandwidths.
        """
        data_support = {variable : (X[variable].min(), X[variable].max()) for variable in X.columns}
        variable_bandwidths = {variable : bw for variable, bw in zip(self.conditionals + self.effects, 
                                                                      self.kYgXZ.bw)}
        support = {}
        for variable in self.conditionals + self.effects:
            K = .01
            if self.variable_types[variable] == 'c':
                lower_support = data_support[variable][0] - K * variable_bandwidths[variable]
                upper_support = data_support[variable][1] + K * variable_bandwidths[variable]
                support[variable] = (lower_support, upper_support)
            else:
                support[variable] = data_support[variable]
        return support    
    
    def CATE(self, *args):
        # takes continuous y, discretes z and x
        data = pd.DataFrame({k : [v] for k, v in zip(self.effects + self.confounders + self.causes, args)})
        pYgXZ = self.kYgXZ.pdf(endog_predict=data[self.effects].values[0], 
                                         exog_predict=data[self.confounders + self.causes].values[0]) 
        
        return data[self.effects].values[0] * pYgXZ

    def ATE(self, x):
        """
        Currently, this does the whole sum/integral over the cube support of Y,
        and consider the confounders set as discrete.
        """
        Ysupp = [self.support[var] for var in self.effects] 

        if self.discrete_Z:
            Zsupp = [range(*(int(self.support[var][0]), int(self.support[var][1])+1)) for var in self.confounders]     
            ATE=[]
            for z in product(*Zsupp):
                z_discrete = pd.DataFrame({k : [v] for k, v in zip(self.confounders, z)})
                z_discrete = z_discrete[self.confounders]
                exog_predictors = x.join(z_discrete)[self.conditionals]
                pZ = self.kZ.pdf(data_predict=z_discrete.values[0])
                ATE.append(nquad(self.CATE, Ysupp, args=tuple(exog_predictors.values[0]))[0] * pZ)
                return sum(ATE)
            
        elif self.continuous_Z:
            supp = [self.support[var] for var in self.effects + self.continuous_Z]
            ATE = nquad(self.CATE, supp, args=tuple(x.values[0]))[0]
            return ATE
        
        else:
            print('no back-door adjustment')
            return nquad(self.CATE, Ysupp, args=tuple(x.values[0]))[0]        
        
    def pdf(self, args):
        """
        Currently, this does the whole sum/integral over the cube support of Z.
        We may be able to improve this by taking into account how the joint
        and conditionals factorize, and/or finding a more efficient support.
        
        This should be reasonably fast for |Z| <= 2 or 3, and small enough discrete
        variable cardinalities.  It runs in O(n_1 n_2 ... n_k) in the cardinality of
        the discrete variables, |Z_1| = n_1, etc.  It likewise runs in O(V^n) for n
        continuous Z variables.  Factorizing the joint/conditional distributions in
        the sum could linearize the runtime.
        """
        args = args[self.causes + self.effects]
        if self.discrete_Z:
            Zsupp = [range(*(int(self.support[var][0]), int(self.support[var][1])+1)) for var in self.discrete_Z]
            pYdoX = []
            for z_vals in product(*Zsupp):
                z_discrete = pd.DataFrame({k : [v] for k, v in zip(self.discrete_Z, z_vals)})
                z_discrete = z_discrete[self.confounders]
                exog_predictors = args.join(z_discrete)[self.conditionals]
                pYgXZ = self.kYgXZ.pdf(exog_predict=exog_predictors, 
                                                           endog_predict=args[self.effects]) 
                pZ = self.kZ.pdf(data_predict=z_discrete)
                pYdoX.append(pYgXZ * pZ)
                
            return sum(pYdoX)
        
        elif self.continuous_Z:
            
            Zsupp = [self.support[var] for var in self.continuous_Z]
            def func(*args):
                data = pd.DataFrame({k : [v] for k, v in zip(self.confounders + self.causes + self.effects, args)})
                pYgXZ = self.kYgXZ.pdf(exog_predict=data[self.conditionals].values[0], 
                                                           endog_predict=data[self.effects].values[0]) 
                pZ = self.kZ.pdf(data_predict=data[self.confounders].values[0])
                return pYgXZ * pZ            
            
            pYdoX, error = nquad(func, Zsupp, args=tuple(args.values[0]))
            return pYdoX 
        
        else:
            return self.kYgXZ.pdf(endog_predict=args[self.effects], exog_predict=args[self.causes])    

        
    def integration_flow(self, *args):
        
        data = pd.DataFrame({k : [v] for k, v in zip(self.effects + self.causes, args)})
        p = self.pdf(data) + 1e-12
        
        if self.discrete_X:
            Xsupp = [range(*(int(self.support[var][0]), int(self.support[var][1])+1)) for var in self.discrete_X]
            q=0.
            for x in product(*Xsupp):
                x_pred = pd.DataFrame({k : [v] for k, v in zip(self.effects + self.causes, args[:-1]+x)})
                x_pred = x_pred[self.effects + self.causes]
                q += self.kX.pdf(data_predict=x_pred[self.causes]) * self.pdf(x_pred)
            q += 1e-12                          
            return - p * np.log(p/q)
        
        else:
            Xsupp = [CE.support[variable] for variable in CE.causes]
            q = nquad(self.pdf, Xsupp, args[-1])[0] 
            q +=1e-12       
        return p * np.log(p/q)
    
    
    def local_flow(self, x):
        """
        Measure the local flow of information from x -> y 
        
        ----------------------------
        INFORMATION FLOW        
                
        The calculation of flow can be broken down into 2 steps. First we generate
        a joint distribution *under interventions*, then we calculate simple the 
        conditional mutual information (CMI) over this distribution. This is described 
        in Ay and Polani's original paper [1] (see Eq. 9), where they say: "It should 
        be noted that the information flow measure can be reformulated in terms of 
        (CMI) with respect to a modified distribution..."
        
        In other words, if we set the joint distribution to:
        
                    p_flow(x, y) := p(x)p(y|do(x))
            
        ... then we can simply measure the CMI:
        
                    I(X -> Y) = I_{pflow}(X ; Y)
        
        Information flow measure can be thought of as the averaged information gathered from
        a intervention experiment.
        
        -----------------------------
        LOCAL FLOW  
        
        Local flow was proposed in Ref.[2] to formalize the notion of specific causal effects
        from an information theoretic perspective.

        It can be defined recursively as the following equation:
        
                  I(X -> Y) = E_{p(x)}[I(x -> Y)],
                  
        where the local flow I(x -> Y) is given by the Kullback–Leibler divergence:

                  I(x -> Y) := D_{KL}[ p(y|do(x)) || E_{p(x')}[p(y|do(x')] ]
        
        the local flow answers the following question: "How much local information would we expect 
        performing the intervention do(X = x) to change the course of nature for Y ?"            
        
        -----------------------------
        [1] https://doi.org/10.1142/S0219525908001465
        [2] https://doi.org/10.3390/e22080854   
        """         
        Ysupp = [self.support[variable] for variable in self.effects]
        
        return - nquad(self.integration_flow, Ysupp, tuple(args.values[0]))[0] 
        
