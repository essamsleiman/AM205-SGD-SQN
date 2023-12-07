import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class StrongConvex(nn.Module):
    def __init__(self, num_dimensions, num_terms):
        super().__init__()
        self.d = num_dimensions
        self.n = num_terms
        self.theta = torch.nn.Parameter(torch.rand(self.d))
        print(f"Initial theta: {self.theta}")

    def forward(self, A, b):
        # A.shape = (n, d, d)
        # b.shape = d
        out = 0
        for i in range(len(A)):
            out += 0.5*self.theta.t()@A[i]@self.theta + b[i].t()@self.theta
        return out

class SQN(torch.optim.Optimizer):
    def __init__(self, params, lr, delta, Gamma, Hessian):
        if delta <= 0.0:
            raise ValueError("Invalid eigenvalue bound.")
        defaults = dict(lr=lr, delta=delta, Gamma=Gamma, Hessian=Hessian)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["Hessian"] = torch.eye(p.numel())
                print(f"Initial Hessian {state['Hessian']}")

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
                
                state = self.state[p]
                # if len(state) == 0:
                #     state["step"] = 0
                state["step"] += 1

                Hessian = state["Hessian"]
                lr = group["lr"]
                delta = group["delta"]
                Gamma = group["Gamma"]

                old_theta = p.data.clone()
                closure()
                old_grad = p.grad.data.clone()

                theta_update = -lr*(torch.inverse(Hessian) + Gamma*torch.eye(Hessian.size(0)))@old_grad
                p.data.add_(theta_update)
                new_theta = p.data

                closure()
                new_grad = p.grad.data
                u = new_theta - old_theta
                v = new_grad - old_grad - delta*u

                Hessian_update = torch.outer(v, v)/torch.dot(u, v) \
                    - torch.outer(torch.mv(Hessian, u), torch.matmul(u.t(), Hessian))/torch.dot(u, torch.mv(Hessian, u)) \
                    + delta*torch.eye(Hessian.size(0))
                Hessian.add_(Hessian_update)

        return loss
    
def get_qf(num_dimension, num_terms, rng):
    A, b = [], []
    for i in range(num_terms):
        A.append(np.diag(rng.uniform(1, 5, size=num_dimension)))
        b.append(rng.uniform(-1, 1, size=num_dimension))
    return np.array(A), np.array(b)

if __name__ == "__main__":

    rng = np.random.default_rng(seed=12345)
    torch.manual_seed(12345)
    d = 50
    N = 1000
    n = 5
    lr = 0.005
    delta = 0.1
    Gamma = 1
    num_epochs = 50

    A, b = get_qf(d, N, rng)
    model = StrongConvex(d, N)
    sqn = SQN(model.parameters(), lr=lr, delta=delta, Gamma=Gamma, Hessian=np.eye(d))
    sgd = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = sqn
    
    losses = []
    for epoch in range(num_epochs):

        idxs = rng.choice(N, n, replace=False)
        A_batch = torch.tensor(A[idxs], dtype=torch.float32)
        b_batch = torch.tensor(b[idxs], dtype=torch.float32)
        
        def closure():
            optimizer.zero_grad()
            loss = model(A_batch, b_batch)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure=closure)
        losses.append(loss.item())

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, num_epochs + 1), losses, "k.-", lw=1, label="RES")
    ax.set_ylabel(r"$f(\theta)$")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig("losses.png", dpi=150)


