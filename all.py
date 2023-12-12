import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import datasets

class StrongConvex(nn.Module):
    def __init__(self, num_dimensions, num_terms):
        super().__init__()
        self.d = num_dimensions
        self.n = num_terms
        self.theta = torch.nn.Parameter(torch.rand(self.d))
        nn.init.constant_(self.theta, 1)

    def forward(self, A, b):
        # A.shape = (n, d, d)
        # b.shape = d
        out = 0
        for i in range(len(A)):
            out += 0.5*self.theta.t()@A[i]@self.theta + b[i].t()@self.theta
        return out/self.n

class SGD(torch.optim.Optimizer):
    """
    Custom PyTorch Opimizer implementing the oBFGS and RES algorithms.
    """
    def __init__(self, params, lr):
        """
        Initializes parameters for the optimizer. These get put into the group dictionary.
        """
        if lr <= 0.0:
            raise ValueError("Invalid learning rate.")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0

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

                # get the update parameters
                lr = group["lr"]

                # compute current gradient
                grad = p.grad.data.clone()

                # update parameter values
                p.data.add_(-lr*grad)
                new_theta = p.data.clone()

        return loss, new_theta.numpy()
    
class SQN(torch.optim.Optimizer):
    """
    Custom PyTorch Opimizer implementing the oBFGS and RES algorithms.
    """
    def __init__(self, params, lr, delta, Gamma, Hessian):
        """
        Initializes parameters for the optimizer. These get put into the group dictionary.
        """
        if delta < 0.0:
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
                theta_update = -lr*(torch.linalg.solve(Hessian, old_grad) + Gamma*torch.eye(Hessian.size(0))@old_grad)
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

        return loss, new_theta.numpy()
    
def get_qf(num_dimension, num_terms, kappa, rng):
    """
    Convenience function to generate the set of matrices and vectors used in the quadratic form of LF1.
    """
    A, b = [], []
    for i in range(num_terms):
        # Choose diagonal A so that we can control conditioning and positive definiteness easily
        size = int(num_dimension/2)
        diag = np.concatenate((rng.uniform(1, 10**(kappa/2), size), \
                               rng.uniform(10**(-kappa/2), 1, size))) 
        A.append(np.diag(diag))
        b.append(rng.uniform(-5, 5, size=num_dimension))
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

def plot_contour(thetas, A, b, label, fig, ax, color):
    # create a contour plot for the 2D case
    x = np.linspace(-1.25, 1.25, 25)
    y = np.linspace(-1.25, 1.25, 25)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, A, b)
    cb = ax.contourf(X, Y, Z, 50, cmap="binary")
    # fig.colorbar(cb)
    ax.plot(thetas[::10, 0], thetas[::10, 1], ".-", lw=1, label=label, color=color)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend()
    # fig.savefig("contour.png", dpi=150)
    return cb

def plot_convergence(losses, solution, label, fig, ax, color):
    errors = losses - solution
    k = 5*np.arange(1, len(errors) + 1)
    ax.semilogy(k, errors, ".-", lw=1, color=color, label=label)
    # ax.loglog([1e2, 1e4], [1e0, 1e-2], "k", lw=3)
    ax.set_xlabel("Function Accesses")
    ax.set_ylabel(r"$f(\theta) - f(\theta^*)$")
    ax.grid()
    ax.legend()
    # fig.savefig("losses.png", dpi=150)

def sc_loss(theta, A, b):
    out = 0
    for j in range(len(A)):
        out += 0.5*theta.T@A[j]@theta + b[j].T@theta
    return out/len(A)

def train(model, optimizer, A, b, num_epochs, N, n, steps=False):
    losses, thetas = [], []
    if steps:
        steps_per_epoch = num_epochs
        num_epochs = 1
    else:
        steps_per_epoch = int(N/n)
    # primary training loop
    for epoch in range(num_epochs):
        for i in range(steps_per_epoch):
            # randomly sample a batch of A and b to compute the stochastic gradients
            idxs = rng.choice(N, n, replace=False)
            A_batch = A[idxs]
            b_batch = b[idxs]
            
            # closure changes at every step because it's defined for a particular batch
            def closure():
                optimizer.zero_grad()
                loss = model(A_batch, b_batch)
                loss.backward()
                return loss
            
            loss_batch, new_theta = optimizer.step(closure=closure)
            # loss_full = model(A, b)
            thetas.append(new_theta)
            losses.append(sc_loss(new_theta, A.numpy(), b.numpy()))
            # commenting this line is all that's needed to remove the effect of the scheduler
            scheduler.step()

            # print loss every few epochs
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_batch.item():.7f}")
    return np.array(losses), np.array(thetas)

if __name__ == "__main__":

    # fix seeds for reproducability
    rng = np.random.default_rng(seed=12345)
    torch.manual_seed(12345)
    d = 2 # problem dimension
    N = 1000 # number of data points
    n = 5 # batch size
    lr = 0.5 # learning rate
    delta = 1e-3 # eigenvalue bound
    Gamma = 1e-4 # constant appearing in Hessian update
    kappa = 5 # controls condition number 10^kappa
    num_epochs = 10

    # obtain sample of A and b
    A, b = get_qf(d, N, kappa, rng)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    
    # get exact solution
    solution, theta = get_exact(A, b)

    labels = ["RES", "SGD", "oBFGS"]
    colors = ["g", "r", "b"]
    fig_contour, ax_contour = plt.subplots()
    fig_converge, ax_converge = plt.subplots()

    for i in range(3):
        print(f"Starting {labels[i]}:")
        model = StrongConvex(d, N)
        if labels[i] == "RES":
            optimizer = SQN(model.parameters(), lr=0.015, delta=delta, Gamma=Gamma, Hessian=np.eye(d))
            scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
            n = 5
        if labels[i] == "oBFGS":
            optimizer = SQN(model.parameters(), lr=0.015, delta=0, Gamma=0, Hessian=np.eye(d))
            scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
            n = 5
        elif labels[i] == "SGD":
            optimizer = SGD(model.parameters(), lr=1.5)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
            n = 1
        losses, thetas = train(model, optimizer, A_tensor, b_tensor, num_epochs, N, n, False)
        # primary plot, shows convergence of excess error vs iteration
        if labels[i] == "SGD":
            losses = losses[::5]
        cb = plot_contour(thetas, A, b, labels[i], fig_contour, ax_contour, colors[i])
        plot_convergence(losses, solution, labels[i], fig_converge, ax_converge, colors[i])

    fig_contour.colorbar(cb)
    ax_contour.set_aspect("equal")
    fig_contour.tight_layout()
    fig_contour.savefig("contour.png", dpi=350)
    fig_converge.tight_layout()
    fig_converge.savefig("converge.png", dpi=350)

