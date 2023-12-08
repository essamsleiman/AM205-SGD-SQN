import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

class StrongConvex(nn.Module):
    def __init__(self, num_dimensions, num_terms):
        super().__init__()
        self.d = num_dimensions
        self.n = num_terms
        self.theta = torch.nn.Parameter(torch.rand(self.d))

    def forward(self, A, b):
        # A.shape = (n, d, d)
        # b.shape = d
        out = 0
        for i in range(len(A)):
            out += 0.5*self.theta.t()@A[i]@self.theta + b[i].t()@self.theta
        return out/self.n

class SQN(torch.optim.Optimizer):
    def __init__(self, params, lr, delta, Gamma, Hessian):
        """
        Initializes parameters for the optimizer. These get put into the group dictionary.
        """
        if delta <= 0.0:
            raise ValueError("Invalid eigenvalue bound.")
        defaults = dict(lr=lr, delta=delta, Gamma=Gamma, Hessian=Hessian)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["Hessian"] = torch.eye(p.numel())

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # state is a dictionary containing the dynamic parameters
                state = self.state[p]
                state["step"] += 1

                # get the Hessian approximation
                Hessian = state["Hessian"]
                # get the update parameters
                lr = group["lr"]
                delta = group["delta"]
                Gamma = group["Gamma"]

                # get the current value of the parameters
                old_theta = p.data.clone()
                # closure zeros the gradient, computes a the loss, and returns it
                closure()
                # compute current gradient
                old_grad = p.grad.data.clone()

                # update parameter values
                theta_update = -lr*(torch.inverse(Hessian) + Gamma*torch.eye(Hessian.size(0)))@old_grad
                p.data.add_(theta_update)
                new_theta = p.data.clone()

                # compute new gradient with SAME batch, NEW parameters
                closure()
                new_grad = p.grad.data
                u = new_theta - old_theta
                v = new_grad - old_grad - delta*u

                # update Hessian approximation, needs optimized
                Hessian_update = torch.outer(v, v)/torch.dot(u, v) \
                    - torch.outer(torch.mv(Hessian, u), torch.matmul(u.t(), Hessian))/torch.dot(u, torch.mv(Hessian, u)) \
                    + delta*torch.eye(Hessian.size(0))
                Hessian.add_(Hessian_update)

        return loss, new_theta
    
def get_qf(num_dimension, num_terms, rng):
    """
    Convenience function to generate the set of matrices and vectors used in the quadratic form of LF1.
    """
    A, b = [], []
    for i in range(num_terms):
        # Choose diagonal A so that we can control conditioning and positive definiteness easily
        A.append(np.diag(rng.uniform(1, 5, size=num_dimension)))
        b.append(rng.uniform(-1, 1, size=num_dimension))
    return np.array(A), np.array(b),

def get_exact(A, b):
    """
    Computes the exact solution to the minimization of LF1 for a given set of A and b
    """
    theta = np.linalg.inv(np.sum(A, axis=0))@np.sum(-b, axis=0)
    out = 0
    for i in range(len(A)):
        out += 0.5*theta.T@A[i]@theta + b[i].T@theta
    return out/len(A), theta

def f(X, Y, A, b):
    """
    Convenience function for computing LF1 on a mesh grid
    """
    out = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            temp_out = 0
            for k in range(len(A)):
                temp_out += 0.5*theta.T@A[k]@theta + b[k].T@theta
            out[i, j] = temp_out
    return out

if __name__ == "__main__":

    # fix seeds for reproducability
    rng = np.random.default_rng(seed=12345)
    torch.manual_seed(12345)
    d = 2 # problem dimension
    N = 1000 # number of data points
    n = 5 # batch size
    lr = 0.01 # learning rate
    delta = 0.1 # eigenvalue bound
    Gamma = 1 # constant appearing in Hessian update
    num_epochs = 5000

    # obtain sample of A and b
    A, b = get_qf(d, N, rng)

    # define the loss function/model
    model = StrongConvex(d, N)
    sqn = SQN(model.parameters(), lr=lr, delta=delta, Gamma=Gamma, Hessian=np.eye(d))
    sgd = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer chooses which algorithm to use
    optimizer = sqn
    # scheduler decreases the learning rate by a fraction gamma every step_size
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    losses = []
    # thetas logs the parameters at every step
    thetas = []
    # primary training loop
    for epoch in range(num_epochs):

        # randomly sample a batch of A and b to compute the stochastic gradients
        idxs = rng.choice(N, n, replace=False)
        A_batch = torch.tensor(A[idxs], dtype=torch.float32)
        b_batch = torch.tensor(b[idxs], dtype=torch.float32)
        
        # closure changes at every step because it's defined for a particular batch
        def closure():
            optimizer.zero_grad()
            loss = model(A_batch, b_batch)
            loss.backward()
            return loss
        
        loss, new_theta = optimizer.step(closure=closure)
        thetas.append(new_theta)
        # commenting this line is all that's needed to remove the effect of the scheduler
        scheduler.step()

        # print loss every few epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}")
    thetas = np.array(thetas)

    # show the exact solution
    solution, theta = get_exact(A, b)
    print(f"Analytical solution: {solution}\n{theta}")

    # create a contour plot for the 2D case
    x = np.linspace(-1, 1, 25)
    y = np.linspace(-1, 1, 25)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, A, b)
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, 50)
    fig.colorbar(contour)
    ax.plot(thetas[::10, 0], thetas[::10, 1], "k.-", lw=1)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    fig.tight_layout()
    fig.savefig("contour.png", dpi=150)

    # computes the true losses given the parameter values at each step
    for i in range(len(thetas)):
        theta = thetas[i]
        out = 0
        for j in range(len(A)):
            out += 0.5*theta.T@A[j]@theta + b[j].T@theta
        losses.append(out/len(A))

    # primary plot, shows convergence of excess error vs iteration
    errors = losses - solution
    k = np.arange(1, num_epochs + 1)
    fig, ax = plt.subplots()
    ax.semilogy(k[::10], errors[::10], "k.-", lw=1, label="RES")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(\theta) - f(\theta^*)$")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig("losses.png", dpi=150)


