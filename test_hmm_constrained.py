import pandas as pd
import numpy as np
from hmmlearn import hmm

# I want a state to be only reached once
# the trick is below :

# ipython -i -m IdxSEC.test_hmm_constrained
if __name__=='__main__':
    n_components = 3
    covariance_type = 'full'
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type,
                            init_params="", params="",
                            n_iter=100)

    model.startprob_ = np.array([1.0, 0.0,0.0])
    ac = 0.995
    acm = 1-ac
    model.transmat_ = np.array([[ac, acm,0.0],
                                [0.0, ac,acm],
                                [0.0, 0.0, 1.0]])
    m1=1.0
    model.means_ = np.array([[0.0],[1.0],[0.0]])
    cov=0.02
    model.covars_=np.array([[[cov]],[[cov]],[[cov]]])

    input_vector = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02, 1.0, 1.01, 1.03, 1.0, 0.0, 0.05, 0.03, 1.0, 1.01, 0.03, 0.04, 0.03])
    input_vector = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02, 1.0, 1.01, 0.03, -0.05, 0.0, 0.05, 0.03, 1.0, 1.01, 0.03, 0.04, 0.03])

    classes_model = model.predict(input_vector.to_frame('e'))
    print(pd.DataFrame({'in':input_vector,'class':classes_model}))
