from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional, KDEMultivariate, EstimatorSettings
import pandas as pd
from itertools import product
from scipy.integrate import nquad
from scipy import stats
import numpy as np

class CausalEffect(object):
    def __init__(self, X, causes, effects, confounders=[], variable_types):
        """
        We want to calculate the causal effect of X and Y through
        back-door adjustment, P(Y|do(X)) = Sum( P(Y|X,Z)P(Z), Z) 
        for some admissable set of control variables, Z.  First we 
        calculate the conditional density P(Y|X,Z), then the density
        P(Z).  We find the support of Z so we can properly sum over
        it later.  variable_types are a dictionary with the column name
        pointing to an element of set(['o', 'u', 'c']), for 'ordered',
        'unordered discrete', or 'continuous'.
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
            density_types = [variable_types[var] for var in confounders]
        else:
            self.variable_types = self.__infer_variable_types(X)

        if 'c' not in variable_types.values():
            bw = 'cv_ml'
        
        else:
            bw = 'normal_reference'

        if confounders:            
            self.density = KDEMultivariate(X[confounders], 
                                  var_type=''.join(density_types),
                                  bw=bw,
                                  defaults=EstimatorSettings(n_jobs=4))
        
        self.conditional_density = KDEMultivariateConditional(endog=X[effects],
                                                         exog=X[conditionals],
                                                         dep_type=''.join(dep_type),
                                                         indep_type=''.join(indep_type),
                                                         bw=bw,
                                                         defaults=EstimatorSettings(n_jobs=4))

        self.support = self.get_support(X)
        
        self.discrete_variables = [variable for variable, var_type in self.variable_types.items() if var_type in ['o', 'u']]
        self.discrete_Z = list(set(self.discrete_variables).intersection(set(confounders)))

        self.continuous_variables = [ variable for variable, var_type in self.variable_types.items() if var_type == 'c' ]
        self.continuous_Z = list(set(self.continuous_variables).intersection(set(confounders)))
 
    def get_support(self, X):
        """
        find the smallest cube around which the densities are supported,
        allowing a little flexibility for variables with small/large bandwidths.
        """
        data_support = {variable : (X[variable].min(), X[variable].max()) for variable in X.columns}
        variable_bandwidths = {variable : bw for variable, bw in zip(self.conditionals + self.effects, 
                                                                      self.conditional_density.bw)}
        print(variable_bandwidths)
        support = {}
        for variable in self.conditionals + self.effects:
            K = 1.
            if self.variable_types[variable] == 'c':
                lower_support = data_support[variable][0] - K * variable_bandwidths[variable]
                upper_support = data_support[variable][1] + K * variable_bandwidths[variable]
                support[variable] = (lower_support, upper_support)
            else:
                support[variable] = data_support[variable]
        return support    
    
        def CATE(self, *args):
            # takes continuous y, discrete z, then x
            data = pd.DataFrame({k : [v] for k, v in zip(self.effect + self.confounders + self.causes, args)})
                                
            pYcond = self.conditional_density.pdf(endog_predict=data[self.effect].values[0], 
                                             exog_predict=data[self.confounders + self.causes].values[0])   

            return data[self.effect].values[0] * pYcond

        def ATE(self, x):
        """
        Currently, this does the whole sum/integral over the cube support of Y,
        and consider the confounders set as discrete.
        """
            Ysupp = [self.support[self.effects]]
            ATE=0.
            if self.discrete_Z:
                
                Zsupp = [range(*(int(self.support[variable][0]), int(self.support[variable][1])+1)) for variable in confounders]     
                for z in product(*Zsupp):
                    z_discrete = pd.DataFrame({k : [v] for k, v in zip(self.confounders, z)})
                    z_discrete = z_discrete[self.confounders]
                    exog_predictors = x.join(z_discrete)[self.conditionals]
                    pZ = self.density.pdf(data_predict=z_discrete.values[0])
                    ATE += nquad(CATE, Ysupp, args=tuple(exog_predictors.values[0])) * pZ
                return ATE[0]   
            else:
                
                return nquad(CATE, Ysupp, args=tuple(exog_predictors.values[0]))[0]         
        
    def integration_flow(self, *args):
        # takes continuous y, discrete z, then x
        data = pd.DataFrame({ k : [v] for k, v in zip(self.effects + self.discrete_Z + self.causes, args)})
        pYcond = self.conditional_density.pdf(exog_predict=data[self.conditionals].values[0], 
                                                   endog_predict=data[self.effects].values[0]) 
        pZ = self.density.pdf(data_predict=data[self.confounders])
        
        return - np.log(pYcond * pZ + 1e-12) * pYcond
    
    
    def local_information_flow(self, x):
        
        """Measure the local flow of information from x -> y 
        
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

                  I(x -> Y) := D_{KL}[ p(y|do(x) || E_{p(x')}[p(y|do(x')] ]
        
        the local flow answers the following question: "How much local information would we expect 
        performing the intervention do(X = x) to change the course of nature for Y ?"            
        
        -----------------------------
        [1] https://doi.org/10.1142/S0219525908001465
        [2] 10.3390/e22080854
        
        """
        pass
