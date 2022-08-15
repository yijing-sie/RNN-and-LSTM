import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r1 = self.Wrx @ x
        self.r2 = self.Wrh @ h
        self.rsum = self.r1 + self.r2 + self.bir + self.bhr
        self.r = self.r_act(self.rsum)
        self.z1 = self.Wzx @ x
        self.z2 = self.Wzh @ h
        self.zsum = self.z1 + self.z2 + self.bhz + self.biz
        self.z = self.z_act(self.zsum)
        self.n1 = self.Wnx @ x
        self.n2 = self.Wnh @ h
        self.n11 = self.n2 + self.bhn
        self.n12 = self.r*self.n11
        self.nsum = self.n12+self.n1+self.bin
        self.n = self.h_act(self.nsum)
        self.h11 = (1-self.z)*self.n
        self.h12 = self.z*h
        h_t = self.h11+self.h12

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.
        

        return h_t
        raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        dz = delta*self.hidden
        dh = delta*self.z
        dn = delta*(1 - self.z)
        dz = dz - delta*self.n
        dn_sum = dn*self.h_act.derivative()
        self.dbin = dn_sum
        dr = dn_sum*self.n11
        dn11 = dn_sum*self.r
        self.dbhn = dn11
        dh = dh + dn11@self.Wnh
        self.dWnh = dn11.reshape(self.h, 1)@self.hidden.reshape(1, self.h)
        self.dWnx = dn_sum.reshape(self.h, 1)@self.x.reshape(1, self.d)
        dx = dn_sum@self.Wnx
        dz_sum = dz*self.z_act.derivative()
        self.dbhz = dz_sum
        self.dbiz = dz_sum
        self.dWzh = dz_sum.reshape(self.h, 1)@self.hidden.reshape(1, self.h)
        dh += dz_sum@self.Wzh
        self.dWzx = dz_sum.reshape(self.h, 1)@self.x.reshape(1, self.d)
        dx += dz_sum@self.Wzx
        dr_sum = dr*self.r_act.derivative()
        self.dbhr = dr_sum
        self.dbir = dr_sum
        self.dWrh = dr_sum.reshape(self.h, 1)@self.hidden.reshape(1, self.h)
        dh += dr_sum@self.Wrh
        self.dWrx = dr_sum.reshape(self.h, 1)@self.x.reshape(1, self.d)
        dx += dr_sum@self.Wrx
        dx = dx.reshape(1, self.d)
        dh.reshape(1, self.h)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        raise NotImplementedError
