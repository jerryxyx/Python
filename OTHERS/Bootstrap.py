import random

def bootstrap(n,n_iter,random_state=None):
    if random_state:
        random.seed(random_state)
    for i in range(n_iter):
        bs=[random.randint(0,n-1) for i in range(n)]
        out_bs=list({i for i in range(n)}-set(bs))
        yield bs,out_bs
