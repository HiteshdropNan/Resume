#!/usr/bin/env python
"""
Forward-Backward Stochastic Neural Networks (FBSNNs) for High-Dimensional PDE & Option Pricing.
Based on Raissi (2018) & Han, Jentzen, E (2018):
Solving High-Dimensional Partial Differential Equations Using Deep Learning.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm


class Sine(nn.Module):
    """Sine activation function module."""
    def forward(self, x):
        return torch.sin(x)


class FCNetwork(nn.Module):
    """Fully connected neural network architecture for value function approximation."""
    def __init__(self, layers, activation="Sine"):
        super().__init__()
        self.net = nn.ModuleList()
        act = Sine() if activation == "Sine" else (nn.ReLU() if activation == "ReLU" else nn.Tanh())
        
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.append(act)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class Naisnet(nn.Module):
    """Non-autonomous neural ODE inspired architecture with skip connections."""
    def __init__(self, layers, activation="Sine"):
        super().__init__()
        self.act = Sine() if activation == "Sine" else (nn.ReLU() if activation == "ReLU" else nn.Tanh())
        self.fc_in = nn.Linear(layers[0], layers[1])
        self.blocks = nn.ModuleList()
        for i in range(1, len(layers) - 2):
            self.blocks.append(nn.Linear(layers[i], layers[i + 1]))
        self.fc_out = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        h = self.act(self.fc_in(x))
        for block in self.blocks:
            h = h + self.act(block(h))
        return self.fc_out(h)


class BlackScholesCallFBSNN:
    """
    Forward-Backward SDE Deep Learning Solver for European Call Options.
    Solves:
      dX_t = r X_t dt + sigma X_t dW_t
      dY_t = r Y_t dt + Z_t dW_t
      Y_T = max(X_T - K, 0)
    where Y_t = u(t, X_t) and Z_t = sigma X_t grad_x u(t, X_t).
    """
    def __init__(self, S0=1.0, K=1.0, T=1.0, r=0.05, sigma=0.25, D=1, N=50, M=64, 
                 layers=None, arch="FC", activation="Sine", device="cpu"):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.D = D
        self.N = N  # Time steps
        self.M = M  # Batch size (trajectories)
        self.dt = T / N
        self.device = torch.device(device)

        if layers is None:
            layers = [D + 1, 128, 128, 128, 1]

        if arch == "Naisnet":
            self.model = Naisnet(layers, activation=activation).to(self.device)
        else:
            self.model = FCNetwork(layers, activation=activation).to(self.device)

        # Initial option price guess Y0 is a trainable parameter
        self.Y0 = nn.Parameter(torch.tensor([0.1], requires_grad=True, device=self.device))
        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.Y0], lr=1e-3)

    def net_u(self, t, X):
        """Neural network approximation of value function u(t, x)."""
        inputs = torch.cat([t, X], dim=1)
        return self.model(inputs)

    def net_z(self, t, X):
        """Computes gradient Z_t = sigma * X * grad_X(u) using automatic differentiation."""
        X_grad = X.clone().detach().requires_grad_(True)
        inputs = torch.cat([t, X_grad], dim=1)
        u = self.model(inputs)
        grad_u = torch.autograd.grad(
            outputs=u, inputs=X_grad,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return self.sigma * X_grad * grad_u

    def forward_simulate(self):
        """Simulates M sample paths over N time steps using Euler-Maruyama scheme."""
        # Initial conditions: shape (M, D)
        X = torch.full((self.M, self.D), self.S0, device=self.device, dtype=torch.float32)
        Y = self.Y0.expand(self.M, 1)

        t_val = 0.0
        dt = self.dt
        sqrt_dt = math.sqrt(dt)

        for n in range(self.N):
            t_tensor = torch.full((self.M, 1), t_val, device=self.device, dtype=torch.float32)
            Z = self.net_z(t_tensor, X)

            # Brownian increment
            dW = torch.randn(self.M, self.D, device=self.device) * sqrt_dt

            # Forward SDE for underlying asset price: dX = r X dt + sigma X dW
            X_next = X + self.r * X * dt + self.sigma * X * dW

            # Backward SDE for option price: dY = r Y dt + Z dW
            Y_next = Y + self.r * Y * dt + torch.sum(Z * dW, dim=1, keepdim=True)

            X = X_next
            Y = Y_next
            t_val += dt

        # Payoff at maturity T
        payoff = torch.clamp(X - self.K, min=0.0)
        loss = torch.mean((Y - payoff) ** 2)
        return loss, self.Y0.item()

    def train_step(self):
        self.optimizer.zero_grad()
        loss, y0 = self.forward_simulate()
        loss.backward()
        self.optimizer.step()
        return loss.item(), y0


def black_scholes_analytical_call(S0, K, T, r, sigma):
    """Closed-form European Call price via standard Black-Scholes formula."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def main():
    print("=" * 70)
    print("Deep FBSDE Solver: High-Dimensional Option Pricing via PyTorch")
    print("=" * 70)

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.25

    bs_price = black_scholes_analytical_call(S0, K, T, r, sigma)
    print(f"Parameters: S0={S0}, K={K}, T={T}y, r={r*100}%, sigma={sigma*100}%")
    print(f"Analytical Black-Scholes Benchmark Price: ${bs_price:.4f}\n")

    # Normalized scale for neural net stability: S0=1.0, K=1.0
    scale = S0
    solver = BlackScholesCallFBSNN(
        S0=1.0, K=1.0, T=T, r=r, sigma=sigma, D=1, N=15, M=32,
        layers=[2, 32, 32, 1], arch="FC", activation="Sine"
    )

    print("Training Forward-Backward Stochastic Neural Network (FBSNN)...", flush=True)
    start_time = time.time()
    iterations = 50

    for it in range(1, iterations + 1):
        loss, y0_norm = solver.train_step()
        current_price = y0_norm * scale

        if it % 10 == 0 or it == 1:
            err = abs(current_price - bs_price)
            print(f"Iter {it:3d}/{iterations} | Loss: {loss:.6f} | Pred Price: ${current_price:.4f} | Abs Error: ${err:.4f}", flush=True)

    elapsed = time.time() - start_time
    final_price = solver.Y0.item() * scale
    final_err_pct = abs(final_price - bs_price) / bs_price * 100

    print("-" * 70, flush=True)
    print(f"Training Completed in {elapsed:.2f} seconds.", flush=True)
    print(f"Final Deep FBSDE Predicted Price: ${final_price:.4f}", flush=True)
    print(f"Exact Black-Scholes Price:       ${bs_price:.4f}", flush=True)
    print(f"Relative Error:                  {final_err_pct:.2f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
