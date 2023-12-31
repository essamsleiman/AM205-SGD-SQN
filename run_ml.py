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
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from torch.nn import init

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def plot_convergence(losses, label, fig, ax, color):
    losses_sliced = losses
    errors = losses_sliced
    k = np.arange(1, args.batch_size * (len(errors))+1, args.batch_size)
    ax.semilogy(k, errors, ".-", lw=1, color=color, label=label)

    # uncomment for 1/k graph
    # if color == 'r':
        # ax.loglog(k[0:3], 1/k[0:3], ".-", lw=1, color='k', label='1/k')
        # ax.semilogy(k, 1/k, ".-", lw=1, color='b', label='1/k')
    ax.set_xlabel("Function Accesses")
    ax.set_ylabel(r"$J_{N}(\theta)$")
    ax.grid('on')
    ax.legend()
    fig.tight_layout()
    fig.savefig("Convex_10epochs_100batch_size_1e-1_lr_final.png", dpi=350)


class SGD_Aaron(torch.optim.Optimizer):
    """
    Custom PyTorch Opimizer implementing the oBFGS and RES algorithms.
    """
    def __init__(self, params, lr):
        """
        Initializes parameters for the optimizer. These get put into the group dictionary.
        """
        print("lr: ", lr)
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

                # get the current value of the parameters
                theta = p.data.clone()
                # compute current gradient
                grad = p.grad.data.clone()

                # update parameter values
                p.data.add_(-lr*grad)
                new_theta = p.data.clone()

        return loss, new_theta.numpy()

class StrongConvex(nn.Module):
    def __init__(self, num_dimensions, num_terms):
        super().__init__()
        self.d = num_dimensions
        self.n = num_terms
        self.theta = torch.nn.Parameter(torch.rand(self.d))

    def forward(self, A, b):
        out = 0
        for i in range(len(A)):
            out += 0.5*self.theta.t()@A[i]@self.theta + b[i].t()@self.theta
        return out/self.n

class SQN(torch.optim.Optimizer):
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

    def stepML1(self, closure=None):
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
                if np.linalg.eigvals(Hessian) < .0001:
                    print("Hessian: ", np.linalg.eigvals(Hessian))
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
                old_grad = old_grad.view(-1) 
                # print("new old_grad: ", old_grad.shape)
                theta_update = -lr*(torch.inverse(Hessian) + Gamma*torch.eye(Hessian.size(0)))@old_grad
                p.data.add_(theta_update)
                new_theta = p.data.clone()

                # compute new gradient with SAME batch, NEW parameters
                closure()
                new_grad = p.grad.data
                u = new_theta - old_theta
                v = new_grad - old_grad - delta*u
                if len(u.size()) > 1:
                    u = u.squeeze(0)
                    v = v.squeeze(0)

                # update Hessian approximation, needs optimized
                Hessian_update = torch.outer(v, v)/torch.dot(u, v) \
                    - torch.outer(torch.mv(Hessian, u), torch.matmul(u.t(), Hessian))/torch.dot(u, torch.mv(Hessian, u)) \
                    + delta*torch.eye(Hessian.size(0))
                Hessian.add_(Hessian_update)

        return loss, new_theta
    def stepML2(self, closure=None):
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
                if np.min(np.linalg.eigvals(Hessian)) < .0001:
                    print("Hessian: ", np.min(np.linalg.eigvals(Hessian)))
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
                # added this line for ML  SQN
                old_grad = old_grad.view(-1) 
                
                theta_update = -lr*(torch.inverse(Hessian) + Gamma*torch.eye(Hessian.size(0)))@old_grad
                theta_update = theta_update.view_as(p.data)

                p.data.add_(theta_update)
                new_theta = p.data.clone()

                # compute new gradient with SAME batch, NEW parameters
                closure()
                new_grad = p.grad.data
                u = new_theta - old_theta

                old_grad = old_grad.view(new_grad.shape)
                v = new_grad - old_grad - delta*u

                # added this line for ML  SQN
                if len(u.size()) > 1:
                    u = u.squeeze(0)
                    v = v.squeeze(0)
                v = v.view(-1)  
                u = u.view(-1)  
                # update Hessian approximation, needs optimized
                Hessian_update = torch.outer(v, v)/torch.dot(u, v) \
                    - torch.outer(torch.mv(Hessian, u), torch.matmul(u.t(), Hessian))/torch.dot(u, torch.mv(Hessian, u)) \
                    + delta*torch.eye(Hessian.size(0))
                # print("Hessian_update: ", Hessian_update.shape)
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


class LogisticRegressionTaskML1(nn.Module):

    def __init__(self, input_dim):
        super(LogisticRegressionTaskML1, self).__init__()
        # Linear layer
        self.linear = nn.Linear(input_dim, 1)
        torch.manual_seed(args.seed)

        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

    
    def forward(self, x):
        # Pass the input through the linear layer, then through sigmoid
        return torch.sigmoid(self.linear(x))
    
    
class LogisticRegressionTaskML2(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LogisticRegressionTaskML2, self).__init__()
        self.linear1 = nn.Linear(input_dim, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 5)
        self.linear4 = nn.Linear(5, 1)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        init.xavier_uniform_(self.linear4.weight)

        # Initialize biases to zero (common practice)
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)
        init.zeros_(self.linear3.bias)
        init.zeros_(self.linear4.bias)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))  # Non-linear activation
        x = F.relu(self.linear2(x))  # Non-linear activation
        x = F.relu(self.linear3(x))  # Non-linear activation
        return torch.sigmoid(self.linear4(x))

def optimize_ML(X, y, model, epochs):
    criterion = nn.BCELoss()
    n_samples, n_features = X.shape
    total_losses = []
    for epoch in range(epochs):
            batch_loss = 0
            for j in range(0, n_samples, args.batch_size):
                # print("j: ", j, n_samples)
                X_batch = torch.tensor(X[j:j+ args.batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y[j:j+ args.batch_size], dtype=torch.float32).unsqueeze(1)

                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

                X_total = torch.tensor(X, dtype=torch.float32).squeeze(-1)
                y_total = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
                total_losses.append(criterion(model(X_total), y_total).item())
            # losses.append(batch_loss)
            print("epoch: ", epoch, "loss: ", sum(total_losses) / len(total_losses))

    return [], total_losses

def optimize_ML_SQN(X, y, thetas, epochs):
    n_samples, n_features = X.shape

    total_losses = []
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        # randomly sample a batch of A and b to compute the stochastic gradients
        def closure():
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            return loss
        batch_loss = 0 
        for j in range(0, n_samples, args.batch_size):

            X_batch = torch.tensor(X[j:j+ args.batch_size], dtype=torch.float32).squeeze(-1)
            y_batch = torch.tensor(y[j:j+ args.batch_size], dtype=torch.float32).unsqueeze(1).squeeze(-1)
            
            if args.model == "ML1":
                loss, new_theta = optimizer.stepML1(closure=closure)
            elif args.model == "ML2" or args.model == "ML3":
                loss ,new_theta = optimizer.stepML2(closure=closure)

            X_total = torch.tensor(X, dtype=torch.float32).squeeze(-1)
            y_total = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            total_losses.append(criterion(model(X_total), y_total).item())
            thetas.append(np.array(new_theta))
            # commenting this line is all that's needed to remove the effect of the scheduler

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {sum(total_losses) / len(total_losses)}")
    return total_losses


def optimize_StrongConvex(A_batch, b_batch, thetas, epochs):
    for epoch in range(epochs):
        # randomly sample a batch of A and b to compute the stochastic gradients
        
        # closure changes at every step because it's defined for a particular batch
        def closure():
            optimizer.zero_grad()
            loss = model(A_batch, b_batch)
            loss.backward()
            return loss
        if args.optimizer == 'SQN':
            loss, new_theta = optimizer.stepML1(closure=closure)
        elif args.optimizer == 'SGD':
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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--delta", type=float, default=1e-4, help="Eigenvalue bound")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Constant in Hessian update")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Gamma for learning rate scheduler")
    parser.add_argument("--num_features", type=int, default=4, help="Number of features for ML problem")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for ML problem")
    parser.add_argument("--model", type=str, default="StrongConvex", choices=["StrongConvex", "ML1", "ML2", "ML3"], help="Model to use (StrongConvex or ML)")
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

    colors = ["g", "r", "b"]
    labels = ["RES", "SGD", "oBFGS"]
    lr_list = [1e-1, 1e-1, 1e-1] # for convex
    # lr_list = [1e-2, 1e-2, 1e-2] # for non-convex

    fig_contour, ax_contour = plt.subplots()
    fig_converge, ax_converge = plt.subplots()
    for ite, optimizer in enumerate(["SQN", "SGD", "SQN"]):
        lr = lr_list[ite]
        args.optimizer = optimizer
        # define the loss function/model
        if args.model == "StrongConvex":
            model = StrongConvex(d, N)
        elif args.model == "ML1":
            model = LogisticRegressionTaskML1(args.num_features)
        elif args.model == "ML2":
            model = LogisticRegressionTaskML2(args.num_features, 8)
        elif args.model == "ML3":
            model = LogisticRegressionTaskML2(args.num_features, 8)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Number of parameters in model: {num_params}")   
        # exit()
        if args.optimizer == "SQN":
            
            if ite == 2:
                optimizer = SQN(model.parameters(), lr=lr, delta=0, Gamma=0, Hessian=np.eye(d))
            else:
                optimizer = SQN(model.parameters(), lr=lr, delta=delta, Gamma=Gamma, Hessian=np.eye(d))
                
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # optimizer = SGD_Aaron(model.parameters(), lr=lr)
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

        elif args.model == "ML1" or args.model == "ML2":
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
            # Convert to binary classification (Iris Setosa vs. others)
            y = (y == 0).astype(int)

            # Feature scaling
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            A_batch = X_train
            b_batch = y_train
        elif args.model == "ML3":
            scaler = StandardScaler()

            df = pd.read_csv('winequality-red.csv')
            X = df.drop(columns= ['quality']).values
            print("x: ", X)
            y = (df['quality'].values > 6).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)  # Use the same scaler for test data
            A_batch = X_train_scaled
            b_batch = y_train

        if args.model == "StrongConvex":
            optimize_StrongConvex(A_batch, b_batch, thetas, args.epochs)
            thetas = np.array(thetas)

        elif args.model == "ML1" or args.model == "ML2" or args.model =="ML3":
            if args.optimizer == "SQN":
                losses = optimize_ML_SQN(A_batch, b_batch, thetas, args.epochs)
                thetas = np.array(thetas)
            else:
                theta, losses = optimize_ML(A_batch, b_batch, model, args.epochs)
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

            plot_convergence(losses, labels[ite], fig_converge, ax_converge, colors[ite])
            