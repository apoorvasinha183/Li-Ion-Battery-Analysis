import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BatteryRNNCell(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, batch_size=1, q_max_base=None, R_0_base=None, D_trainable=False):
        super(BatteryRNNCell, self).__init__()

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.q_max_base_value = q_max_base
        self.R_0_base_value = R_0_base
        self.curr_cum_pwh = curr_cum_pwh

        self.q_max_model = q_max_model
        self.R_0_model = R_0_model

        self.state_size = 8
        self.output_size = 1
        
        self.MLPp = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

        self.MLPn = nn.Sequential(
            nn.Linear(1, 1)
        )

        # Initialize MLPp weights
        # Load the weights from the .pth file
        weights_path = 'mlp_initial_weights.pth'
        mlp_p_weights = torch.load(weights_path)

        # Assuming self.MLPp is your PyTorch Sequential model
        with torch.no_grad():
            # Assign weights and biases to each layer in the model
            self.MLPp[0].weight.copy_(mlp_p_weights['0.weight'])
            self.MLPp[0].bias.copy_(mlp_p_weights['0.bias'])
            self.MLPp[2].weight.copy_(mlp_p_weights['2.weight'])
            self.MLPp[2].bias.copy_(mlp_p_weights['2.bias'])
            self.MLPp[4].weight.copy_(mlp_p_weights['4.weight'])
            self.MLPp[4].bias.copy_(mlp_p_weights['4.bias'])

        # Initialize MLPn weights
        X = torch.linspace(0.0, 1.0, 100).unsqueeze(1)
        Y = torch.linspace(-8e-4, 8e-4, 100).unsqueeze(1)
        self.MLPn_optim = torch.optim.Adam(self.MLPn.parameters(), lr=2e-2)
        for _ in range(200):
            self.MLPn_optim.zero_grad()
            output = self.MLPn(X)
            loss = F.mse_loss(output, Y)
            loss.backward()
            self.MLPn_optim.step()

        for param in self.MLPn.parameters():
            param.requires_grad = False

    def initBatteryParams(self, batch_size, D_trainable):
        self.q_max_base_value = 1.0e4 if self.q_max_base_value is None else self.q_max_base_value
        self.R_0_base_value = 1.0e1 if self.R_0_base_value is None else self.R_0_base_value

        self.xnMax = torch.tensor(0.6)
        self.xnMin = torch.tensor(0.0)
        self.xpMax = torch.tensor(1.0)
        self.xpMin = torch.tensor(0.4)

        self.qMaxBASE = torch.tensor(self.q_max_base_value)
        self.RoBASE = torch.tensor(self.R_0_base_value)

        if self.q_max_model is None:
            initial_q_max = torch.tensor(1.4e4 / self.q_max_base_value)
            self.qMax = nn.Parameter(initial_q_max * torch.ones(batch_size))
        else:
            self.qMax = self.q_max_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.qMaxBASE

        if self.R_0_model is None:
            initial_R_0 = torch.tensor(0.15 / self.R_0_base_value)
            self.Ro = nn.Parameter(initial_R_0 * torch.ones(batch_size))
        else:
            self.Ro = self.R_0_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.RoBASE

        self.R = torch.tensor(8.3144621)
        self.F = torch.tensor(96487)
        self.alpha = torch.tensor(0.5)
        self.VolSFraction = torch.tensor(0.1)
        self.Sn = torch.tensor(2e-4)
        self.Sp = torch.tensor(2e-4)
        self.kn = torch.tensor(2e4)
        self.kp = torch.tensor(2e4)
        self.Vol = torch.tensor(2.2e-5)

        self.VolS = self.VolSFraction * self.Vol
        self.VolB = self.Vol - self.VolS

        self.qpMin = self.qMax * self.qMaxBASE * self.xpMin
        self.qpMax = self.qMax * self.qMaxBASE * self.xpMax
        self.qpSMin = self.qpMin * self.VolS / self.Vol
        self.qpBMin = self.qpMin * self.VolB / self.Vol
        self.qpSMax = self.qpMax * self.VolS / self.Vol
        self.qpBMax = self.qpMax * self.VolB / self.Vol
        self.qnMin = self.qMax * self.qMaxBASE * self.xnMin
        self.qnMax = self.qMax * self.qMaxBASE * self.xnMax
        self.qnSMax = self.qnMax * self.VolS / self.Vol
        self.qnBMax = self.qnMax * self.VolB / self.Vol
        self.qnSMin = self.qnMin * self.VolS / self.Vol
        self.qnBMin = self.qnMin * self.VolB / self.Vol
        self.qSMax = self.qMax * self.qMaxBASE * self.VolS / self.Vol
        self.qBMax = self.qMax * self.qMaxBASE * self.VolB / self.Vol

        self.to = torch.tensor(10.0)
        self.tsn = torch.tensor(90.0)
        self.tsp = torch.tensor(90.0)
        self.U0p = torch.tensor(4.03)
        self.U0n = torch.tensor(0.01)
        self.VEOD = torch.tensor(3.0)

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_state(batch_size=inputs.shape[0])

        next_states = self.getNextState(states, inputs)
        output = self.getNextOutput(next_states, inputs)
        return output, next_states

    def getNextOutput(self, X, U):
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = X.split(1, dim=1)
        i = U

        qSMax = (self.qMax * self.qMaxBASE) * self.VolS / self.Vol
        Tbm = Tb - 273.15
        xpS = qpS / qSMax
        xnS = qnS / qSMax

        VepMLP = self.MLPp(xpS)
        VenMLP = self.MLPn(xnS)

        safe_log_p = torch.clamp((1 - xpS) / xpS, 1e-18, 1e+18)
        safe_log_n = torch.clamp((1 - xnS) / xnS, 1e-18, 1e+18)

        Vep = self.U0p + self.R * Tb / self.F * torch.log(safe_log_p) + VepMLP
        Ven = self.U0n + self.R * Tb / self.F * torch.log(safe_log_n) + VenMLP
        V = Vep - Ven - Vo - Vsn - Vsp

        return V

    def getNextState(self, X, U):
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = X.split(1, dim=1)
        i = U

        qSMax = (self.qMax * self.qMaxBASE) * self.VolS / self.Vol
        xpS = torch.clamp(qpS / qSMax, 1e-18, 1.0)
        xnS = torch.clamp(qnS / qSMax, 1e-18, 1.0)
        Jn0 = 1e-18 + self.kn * (1 - xnS) ** self.alpha * (xnS) ** self.alpha
        Jp0 = 1e-18 + self.kp * (1 - xpS) ** self.alpha * (xpS) ** self.alpha

        Tbdot = torch.zeros_like(Tb)
        CnBulk = qnB / self.VolB
        CnSurface = qnS / self.VolS
        CpSurface = qpS / self.VolS
        CpBulk = qpB / self.VolB
        qdotDiffusionBSn = (CnBulk - CnSurface) / self.tDiffusion
        qnBdot = -qdotDiffusionBSn
        qdotDiffusionBSp = (CpBulk - CpSurface) / self.tDiffusion
        qpBdot = -qdotDiffusionBSp
        qpSdot = i + qdotDiffusionBSp
        Jn = i / self.Sn
        VoNominal = i * self.Ro * self.RoBASE
        Jp = i / self.Sp
        qnSdot = qdotDiffusionBSn - i
        VsnNominal = self.R * Tb / self.F / self.alpha * torch.asinh(Jn / (2 * Jn0))
        Vodot = (VoNominal - Vo) / self.to
        VspNominal = self.R * Tb / self.F / self.alpha * torch.asinh(Jp / (2 * Jp0))
        Vsndot = (VsnNominal - Vsn) / self.tsn
        Vspdot = (VspNominal - Vsp) / self.tsp

        dt = self.dt
        XNew = torch.cat([
            Tb + Tbdot * dt,
            Vo + Vodot * dt,
            Vsn + Vsndot * dt,
            Vsp + Vspdot * dt,
            qnB + qnBdot * dt,
            qnS + qnSdot * dt,
            qpB + qpBdot * dt,
            qpS + qpSdot * dt
        ], dim=1)

        return XNew

    def get_initial_state(self, batch_size=None):
        self.initBatteryParams(batch_size, D_trainable=False)
        if self.q_max_model is not None:
            qMax = torch.cat([self.q_max_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.qMaxBASE for _ in range(batch_size)], dim=0)
        else:
            qMax = self.qMax

        if self.R_0_model is not None:
            Ro = torch.cat([self.R_0_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.RoBASE for _ in range(batch_size)], dim=0)
        else:
            Ro = self.Ro

        qpMin = qMax * self.qMaxBASE * self.xpMin
        qpSMin = qpMin * self.VolS / self.Vol
        qpBMin = qpMin * self.VolB / self.Vol
        qnMax = qMax * self.qMaxBASE * self.xnMax
        qnSMax = qnMax * self.VolS / self.Vol
        qnBMax = qnMax * self.VolB / self.Vol

        if self.initial_state is None:
            initial_state = torch.cat([
                292.1 * torch.ones(batch_size, 1),
                torch.zeros(batch_size, 3),
                qnBMax.view(batch_size, 1),
                qnSMax.view(batch_size, 1),
                qpBMin.view(batch_size, 1),
                qpSMin.view(batch_size, 1)
            ], dim=1)
        else:
            initial_state = torch.tensor(self.initial_state)

        return initial_state
