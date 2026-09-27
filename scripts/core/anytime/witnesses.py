"""Predictors train only AFTER both scores in each pair have been recorded."""
import numpy as np

def softmax(x):
    z=x-x.max(axis=1,keepdims=True);p=np.exp(z);return .01/p.shape[1]+.99*p/p.sum(axis=1,keepdims=True)

class RunningRidge:
    def __init__(self,dimension,classes=2):
        self.classes=classes;self.inverse=np.eye(dimension+1);self.xy=np.zeros(dimension+1);self.n_seen=0
    def features(self,x):return np.column_stack((np.ones(len(x)),np.tanh(np.asarray(x,float)/3.)))
    def predict(self,x):
        mean=(self.features(x)@self.inverse@self.xy)*(self.classes-1)
        logits=-(np.arange(self.classes)[None,:]-mean[:,None])**2/2
        return softmax(logits)
    def learn(self,x,y):
        for a,b in zip(self.features(x),y):
            v=self.inverse@a;self.inverse-=np.outer(v,v)/(1+a@v);self.xy+=a*(b/(self.classes-1));self.n_seen+=1

class GeometricMLP:
    def __init__(self,dimension,classes=2,seed=7):
        self.classes=classes;self.n_seen=0;self.next_fit=32;self.x=[];self.y=[]
        rng=np.random.default_rng(seed);self.w=rng.normal(scale=.3,size=(dimension,8));self.b=rng.normal(scale=.2,size=8);self.v=rng.normal(scale=.2,size=(8,classes));self.c=np.zeros(classes)
    def predict(self,x):return softmax(np.tanh(np.tanh(np.asarray(x,float)/3.)@self.w+self.b)@self.v+self.c)
    def learn(self,x,y):
        self.x.extend(np.asarray(x,float));self.y.extend(map(int,y));self.n_seen+=len(y)
        if self.n_seen<self.next_fit:return
        a=np.tanh(np.asarray(self.x)/3.);y=np.asarray(self.y);one=np.eye(self.classes)[y]
        for _ in range(30):
            h=np.tanh(a@self.w+self.b);p=softmax(h@self.v+self.c);err=(p-one)/len(y);back=(err@self.v.T)*(1-h*h)
            self.v-=.3*(h.T@err+.001*self.v);self.c-=.3*err.sum(0);self.w-=.3*(a.T@back+.001*self.w);self.b-=.3*back.sum(0)
        self.next_fit*=2

class PredictableProduct:
    def __init__(self):self.n=0;self.total=np.zeros(2)
    def transform(self,x):
        mean=self.total/max(self.n,1)
        return ((np.asarray(x)[:,:2]-mean).prod(axis=1))[:,None]
    def learn(self,x):self.total+=np.asarray(x)[:,:2].sum(0);self.n+=len(x)

class PairEngine:
    """Auditable predict/bet/learn ordering; current labels never passed to predictor."""
    def __init__(self,witness,process):self.witness=witness;self.process=process
    def step(self,x,y):
        before=self.witness.n_seen;p=self.witness.predict(x)
        if self.witness.n_seen!=before:raise RuntimeError('prediction mutated training state')
        inc=self.process.update(p,y);self.witness.learn(x,y);return inc
