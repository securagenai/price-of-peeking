"""Log-domain e-processes under explicit conditional label/swap nulls."""
import math
import numpy as np

class PredictableFraction:
    """Regularized past-score plug-in bet; not claimed to reproduce ONS/aGRAPA."""
    def __init__(self):self.total=0.;self.squares=1.
    def value(self):return float(np.clip(self.total/self.squares,0.,.5))
    def update(self,d):self.total+=float(d);self.squares+=float(d)**2

def swap_score(probabilities,labels):
    p=np.asarray(probabilities,float);y=np.asarray(labels,int)
    if p.ndim!=2 or len(p)!=2 or y.shape!=(2,) or not np.all(np.isfinite(p)):raise ValueError('pair required')
    if np.any(p<0) or not np.allclose(p.sum(1),1.):raise ValueError('proper probabilities required')
    d=(p[0,y[0]]+p[1,y[1]]-p[0,y[1]]-p[1,y[0]])/2
    if abs(d)>1+1e-12:raise ArithmeticError('unbounded payoff')
    return float(d)

class SwapProcess:
    def __init__(self):self.bet=PredictableFraction();self.log_wealth=0.;self.steps=0
    def update(self,p,y):
        d=swap_score(p,y);fraction=self.bet.value() # chosen BEFORE current score
        increment=math.log1p(fraction*d)
        self.log_wealth+=increment;self.steps+=1;self.bet.update(d)
        return increment

class LikelihoodProcess:
    def __init__(self,p):
        self.p=np.asarray(p,float)
        if self.p.ndim!=1 or np.any(self.p<=0) or not np.isclose(self.p.sum(),1):raise ValueError('known positive marginal required')
        self.log_wealth=0.;self.steps=0
    def update(self,q,y):
        q=np.asarray(q,float);y=np.asarray(y,int)
        if q.shape!=(len(y),len(self.p)) or np.any(q<=0) or not np.all(np.isfinite(q)) or not np.allclose(q.sum(1),1):raise ValueError('strict positive normalized q required')
        inc=np.log(q[np.arange(len(y)),y])-np.log(self.p[y]);self.log_wealth+=float(inc.sum());self.steps+=len(y)
        return inc

def mixture_logwealth(logwealth):
    a=np.asarray(logwealth,float);m=float(np.max(a))
    return m+math.log(float(np.exp(a-m).mean()))

def crossed(logwealth,alpha):
    if not 0<alpha<1:raise ValueError('alpha')
    return bool(logwealth>=math.log(1/alpha))

def e_bh(log_evalues,alpha):
    """One fixed/common stopping time; not a claim of continuous FDR across times."""
    a=np.asarray(log_evalues);order=np.argsort(-a);m=len(a);k=0
    for i,j in enumerate(order,1):
        if a[j]>=math.log(m/(alpha*i)):k=i
    return sorted(map(int,order[:k]))

class BettingLogGrowthCS:
    """Hoeffding exponential-betting CS for time-average conditional log growth.

    Uses delta/(n(n+1)) time allocation. Works for adapted bounded increments,
    not only IID increments. This is not automatically a mutual-information CS.
    """
    def __init__(self,lower,upper,delta=.05):
        if not lower<upper or not 0<delta<1:raise ValueError('bounds/delta')
        self.lower=lower;self.upper=upper;self.delta=delta;self.total=0.;self.n=0
    def update(self,value):
        if not math.isfinite(value) or not self.lower-1e-12<=value<=self.upper+1e-12:raise ValueError('increment outside declared bounds')
        self.n+=1;self.total+=value
        radius=(self.upper-self.lower)*math.sqrt(math.log(self.n*(self.n+1)/self.delta)/(2*self.n))
        return self.total/self.n-radius

def heuristic_tau(pi,alpha):
    return None if not np.isfinite(pi) or pi<=0 else math.log(1/alpha)/pi
