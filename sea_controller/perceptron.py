import numpy as np



def sigmoid(z):
    return 1/(1+np.exp(-z))

def d_sigmoid(z):
    return np.multiply((sigmoid(z)),(1-sigmoid(z)))


def ReLU(z):
    return np.maximum(z,0)

def d_ReLU(z):
    x = z[:]
    x[x<=0] = 0
    x[x>0] = 1
    return x

def tanh(z):
    return np.tanh(z/2.0)

def d_tanh(z):
    return (np.cosh(z)+1)

def cube_root(z):
    x = z[:]
    l = [0.5*a[0,0] if a[0,0]>-2 and a[0,0]<2 else np.cbrt(a[0,0]) for a in x]
    l = np.matrix([l]).T
    return l

def d_cube_root(z):
    x = z[:]
    return np.matrix([[0.5 if a[0,0]>-2 and a[0,0]<2 else (np.cbrt(1.0/np.square(a[0,0])))/3.0 for a in x]]).T

def leaky_ReLU(z):
    x = z[:]
    l = [a[0,0] if a[0,0] >= 0 else 0.3*a[0,0] for a in x]
    return np.matrix([l]).T

def d_leaky_ReLU(z):
    x = z[:]
    l = [1 if a[0,0] >= 0 else 0.3 for a in x]
    return np.matrix([l]).T

def linear(z):
    return np.maximum(np.minimum(z,50),-50)

def d_linear(z):
    x = z[:]
    x[x>50] = 0.001
    x[x<-50] = 0.001
    return x

def linear_unfiltered(z):
    return z

def d_linear_unfiltered(z):
    return np.ones(z.shape)


def diff(y,y_pred):
    return y-y_pred


## input:
##  [
##      e1
##      e2
##      ...
##      en
##  ]
##
## weights:
##  [
##      w11 w12 w13 ... w1n
##      w21 w22 w23 ... w2n
##      ...
##      wm1 wm2 wm3 ... wmn
##  ]
##




class Perceiver:

    def __init__(self, layer_info, activation=None, d_activation=None, loss=diff, lr=0.05):
        if activation==None and d_activation==None:
            self.activation=[tanh]*(len(layer_info)+1)
            self.d_activation=[d_tanh]*(len(layer_info)+1)
        else:
            self.activation=activation
            self.d_activation=d_activation

        self.loss=loss

        self.lr=lr

        self.w_b = []
        for l in layer_info:
            in_dims = l[0]
            out_dims = l[1]
            self.w_b.append([20*np.random.random_sample((out_dims,in_dims)), np.random.random_sample((out_dims,1))])


    # forward prop
    def predict(self, X):
        # x = np.matrix(X).T
        x = X

        A = []

        i = 0
        for wb in self.w_b:
            w = wb[0]
            b = wb[1]

            a = self.activation[i](x)
            A.append(a)

            x = np.add(np.dot(w,a),b)
            i += 1

        return self.activation[i](x), A

    # backprop
    def update(self, predicted, Y, A):

        w_grads = []
        b_grads = []

        error = np.multiply(self.loss(Y,predicted),self.d_activation[-1](predicted))
        w_grad = np.multiply(A[-1].T, error)
        b_grad = error

        w_grads.append(self.lr*w_grad)
        b_grads.append(self.lr*b_grad)

        for i in range(len(self.w_b)-1, 0, -1):

            error = np.multiply(np.dot(self.w_b[i][0].T,error), self.d_activation[i](A[i]))

            w_grad = np.multiply(A[i-1].T, error)
            b_grad = error

            w_grads.append(self.lr*w_grad)
            b_grads.append(self.lr*b_grad)

        return w_grads, b_grads


    def train(self, X, Y, epochs=5, batch_size=100):

        for _ in range(epochs):

            cumm_w_grads = [0]*len(self.w_b)
            cumm_b_grads = [0]*len(self.w_b)

            elems = X.shape[1]

            elem_cnt = 0

            for i in range(elems):
                elem_cnt += 1
                predicted, A = self.predict(X[:,i])
                wg, bg = self.update(predicted,Y[:,i],A)

                j = 0
                for w in reversed(wg):
                    cumm_w_grads[j] += w
                    j += 1
                j = 0
                for b in reversed(bg):
                    cumm_b_grads[j] += b
                    j += 1


                if elem_cnt == batch_size:

                    for k in range(len(cumm_w_grads)):
                        self.w_b[k][0] += cumm_w_grads[k]/batch_size
                        self.w_b[k][1] += cumm_b_grads[k]/batch_size

                    cumm_w_grads = [0]*len(self.w_b)
                    cumm_b_grads = [0]*len(self.w_b)

                    elem_cnt = 0



