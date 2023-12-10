import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        print("here")
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
                print("Hessian: ", Hessian.shape)
                print("torch.eye(Hessian.size(0))): ", torch.eye(Hessian.size(0)).size())
                print("old_grad: ", old_grad.shape)
                old_grad = old_grad[0]
                # print("new old_grad: ", old_grad.shape)
                theta_update = -lr*(torch.inverse(Hessian) + Gamma*torch.eye(Hessian.size(0)))@old_grad
                p.data.add_(theta_update)
                new_theta = p.data.clone()

                # compute new gradient with SAME batch, NEW parameters
                closure()
                new_grad = p.grad.data
                u = new_theta - old_theta
                v = new_grad - old_grad - delta*u
                print("v:", v.shape)

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


class LogisticRegressionTask(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionTask, self).__init__()
        # Linear layer
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Pass the input through the linear layer, then through sigmoid
        return torch.sigmoid(self.linear(x))

def optimize_ML(X, y, model, epochs):
    criterion = nn.BCELoss()
    n_samples, n_features = X.shape
    # theta = np.zeros(n_features)
    losses = []
    for epoch in range(epochs):
            for j in range(0, n_samples, args.batch_size):
                X_batch = torch.tensor(X[j:j+ args.batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y[j:j+ args.batch_size], dtype=torch.float32).unsqueeze(1)

                # Forward pass
                # print('X_batch: ', X_batch.size())
                # print('model: ', model)
                outputs = model(X_batch)
                print("outputs", outputs.shape)
                

                loss = criterion(outputs, y_batch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            print("epoch: ", epoch, "loss: ", sum(losses))

    with torch.no_grad():
        theta = model.linear.weight.detach().numpy().flatten()
    print('theta: ', theta)
    return theta, losses

def optimize_ML_SQN(X, y, thetas, epochs):
    n_samples, n_features = X.shape
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        # randomly sample a batch of A and b to compute the stochastic gradients
        def closure():
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            # print("outputs: ", outputs.size())
            # print("outputs: ", outputs.squeeze(-1).size())
            loss = criterion(outputs, y_batch)
            # loss = model(A_batch, b_batch)
            loss.backward()
            return loss
        
        for j in range(0, n_samples, args.batch_size):
            X_batch = torch.tensor(X[j:j+ args.batch_size], dtype=torch.float32).squeeze(-1)
            y_batch = torch.tensor(y[j:j+ args.batch_size], dtype=torch.float32).unsqueeze(1).squeeze(-1)
            print("X_batch", X_batch.shape)
            print("y_batch", y_batch.shape)

            loss, new_theta = optimizer.step(closure=closure)
            thetas.append(np.array(new_theta))
            # commenting this line is all that's needed to remove the effect of the scheduler
            scheduler.step()

        # print loss every few epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}")
    return thetas


def optimize_StrongConvex(A_batch, b_batch, thetas, epochs):
    for epoch in range(epochs):
        # randomly sample a batch of A and b to compute the stochastic gradients
        
        # closure changes at every step because it's defined for a particular batch
        def closure():
            optimizer.zero_grad()
            loss = model(A_batch, b_batch)
            loss.backward()
            return loss
        
        loss, new_theta = optimizer.step(closure=closure)
        thetas.append(np.array(new_theta))
        # commenting this line is all that's needed to remove the effect of the scheduler
        scheduler.step()

        # print loss every few epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}")
    return thetas

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Command line arguments for training")

    # Add arguments
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--dim", type=int, default=2, help="Problem dimension")
    parser.add_argument("--data_points", type=int, default=1000, help="Number of data points")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--delta", type=float, default=0.1, help="Eigenvalue bound")
    parser.add_argument("--gamma", type=float, default=1, help="Constant in Hessian update")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Gamma for learning rate scheduler")
    parser.add_argument("--num_features", type=int, default=4, help="Number of features for ML problem")
    parser.add_argument("--model", type=str, default="StrongConvex", choices=["StrongConvex", "ML"], help="Model to use (StrongConvex or ML)")
    parser.add_argument("--optimizer", type=str, default="SQN", choices=["SQN", "SGD"], help="Optimizer to use (SQN or SGD)")


    # Parse the arguments
    args = parser.parse_args()

    # fix seeds for reproducability
    rng = np.random.default_rng(seed=args.seed)
    torch.manual_seed(args.seed)
    d = args.dim # problem dimension
    N = args.data_points # number of data points
    n = args.batch_size # batch size
    lr = args.lr # learning rate
    delta = args.delta # eigenvalue bound
    Gamma = args.gamma # constant appearing in Hessian update
    num_epochs = args.epochs

    # obtain sample of A and b
    A, b = get_qf(d, N, rng)

    # define the loss function/model
    if args.model == "StrongConvex":
        model = StrongConvex(d, N)
    elif args.model == "ML":
        model = LogisticRegressionTask(args.num_features)
    else:
        model = None


    if args.optimizer == "SQN":
        optimizer = SQN(model.parameters(), lr=lr, delta=delta, Gamma=Gamma, Hessian=np.eye(d))
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = None

    # optimizer chooses which algorithm to use
    # scheduler decreases the learning rate by a fraction gamma every step_size
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)
    
    losses = []
    # thetas logs the parameters at every step
    thetas = []
    # primary training loop

    # Data loading
    if args.model == "StrongConvex":
        idxs = rng.choice(N, n, replace=False)
        A_batch = torch.tensor(A[idxs], dtype=torch.float32)
        b_batch = torch.tensor(b[idxs], dtype=torch.float32)
    elif args.model == "ML":
        iris = datasets.load_iris()
        X = iris.data  # Features
        y = iris.target  # Labels

        # Convert to binary classification (Iris Setosa vs. others)
        y = (y == 0).astype(int)

        # Feature scaling
        print("x: ",X.shape)
        print("y: ",y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        A_batch = X_train
        b_batch = y_train


    if args.model == "StrongConvex":
        optimize_StrongConvex(A_batch, b_batch, thetas, args.epochs)
        thetas = np.array(thetas)

    elif args.model == "ML":
        if args.optimizer == "SQN":
            optimize_ML_SQN(A_batch, b_batch, thetas, args.epochs)
            thetas = np.array(thetas)
        else:
            theta, losses = optimize_ML(model, args.epochs)
            thetas = theta

        
    thetas = np.array(thetas)
    print("thetas: ", thetas.shape)
    # show the exact solution
    solution, theta = get_exact(A, b)
    print(f"Analytical solution: {solution}\n{theta}")

    # create a contour plot for the 2D case
    if args.model == "StrongConvex":
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

    else:
        k = np.arange(1, args.epochs + 1)
        fig, ax = plt.subplots()
        print("k: ", k)
        print("losses: ", losses)
        ax.semilogy(k, losses, "k.-", lw=1, label="RES")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$f(\theta) - f(\theta^*)$")
        ax.grid()
        ax.legend()
        fig.tight_layout()
        fig.savefig("losses.png", dpi=150)
