import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BatteryRNNCell(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, q_max_base=None, R_0_base=None, D_trainable=False):
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
        self.initBatteryParams(D_trainable)

        self.MLPp = nn.Sequential(
            nn.Flatten(),
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
        weights_path = 'torch_train/mlp_initial_weights.pth'
        mlp_p_weights = torch.load(weights_path)
        #for keys in mlp_p_weights["model_state_dict"]:
        #    print("keys available are ",keys)
        #self.MLPp.load_state_dict(mlp_p_weights['model_state_dict'])
        #self.MLPp.train()
        with torch.no_grad():
            # Assign weights and biases to each layer in the model
            self.MLPp[1].weight.copy_(mlp_p_weights["model_state_dict"][ "MLPp.1.weight"])
            self.MLPp[1].bias.copy_(mlp_p_weights["model_state_dict"]['MLPp.1.bias'])
            self.MLPp[3].weight.copy_(mlp_p_weights["model_state_dict"]['MLPp.3.weight'])
            self.MLPp[3].bias.copy_(mlp_p_weights["model_state_dict"]['MLPp.3.bias'])
            self.MLPp[5].weight.copy_(mlp_p_weights["model_state_dict"]['MLPp.5.weight'])
            self.MLPp[5].bias.copy_(mlp_p_weights["model_state_dict"]['MLPp.5.bias'])
        # print("Success!")
        # Initialize MLPn weights
        self.MLPp.to(DEVICE)
        X = torch.linspace(0.0, 1.0, 100).unsqueeze(1).to(DEVICE)

        Y = torch.linspace(-8e-4, 8e-4, 100).unsqueeze(1).to(DEVICE)
        self.MLPn.to(DEVICE)
        #This is such a chad move
        self.MLPn_optim = torch.optim.Adam(self.MLPn.parameters(), lr=2e-2)
        for _ in range(200):
            self.MLPn_optim.zero_grad()
            output = self.MLPn(X)
            loss = F.mse_loss(output, Y)
            loss.backward()
            self.MLPn_optim.step()

        for param in self.MLPn.parameters():
            param.to(DEVICE)
            param.requires_grad = False
        self.MLPn.to(DEVICE)    

    def initBatteryParams(self,D_trainable):
        self.q_max_base_value = 1.0e4 if self.q_max_base_value is None else self.q_max_base_value
        self.R_0_base_value = 1.0e1 if self.R_0_base_value is None else self.R_0_base_value

        self.xnMax = torch.tensor(0.6).to(DEVICE)
        self.xnMin = torch.tensor(0.0).to(DEVICE)
        self.xpMax = torch.tensor(1.0).to(DEVICE)
        self.xpMin = torch.tensor(0.4).to(DEVICE)
        if not D_trainable:
            self.tDiffusion = torch.tensor(7e6).to(DEVICE)
        self.qMaxBASE = torch.tensor(self.q_max_base_value).to(DEVICE)
        self.RoBASE = torch.tensor(self.R_0_base_value).to(DEVICE)

        if self.q_max_model is None:
            initial_q_max = torch.tensor(1.4e4 / self.q_max_base_value).to(DEVICE)
            self.qMax = nn.Parameter(initial_q_max).to(DEVICE)
        else:
            self.qMax = self.q_max_model(torch.tensor(self.curr_cum_pwh)) / self.qMaxBASE
            self.qMax.to(DEVICE)

        if self.R_0_model is None:
            initial_R_0 = torch.tensor(0.15 / self.R_0_base_value)
            self.Ro = nn.Parameter(initial_R_0).to(DEVICE)
        else:
            self.Ro = self.R_0_model(torch.tensor(self.curr_cum_pwh)) / self.RoBASE
            self.Ro.to(DEVICE)

        self.R = torch.tensor(8.3144621).to(DEVICE)
        self.F = torch.tensor(96487).to(DEVICE)
        self.alpha = torch.tensor(0.5).to(DEVICE)
        self.VolSFraction = torch.tensor(0.1).to(DEVICE)
        self.Sn = torch.tensor(2e-4).to(DEVICE)
        self.Sp = torch.tensor(2e-4).to(DEVICE)
        self.kn = torch.tensor(2e4).to(DEVICE)
        self.kp = torch.tensor(2e4).to(DEVICE)
        self.Vol = torch.tensor(2.2e-5).to(DEVICE)

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

        self.t0 = torch.tensor(10.0).to(DEVICE)
        self.tsn = torch.tensor(90.0).to(DEVICE)
        self.tsp = torch.tensor(90.0).to(DEVICE)
        self.U0p = torch.tensor(4.03).to(DEVICE)
        self.U0n = torch.tensor(0.01).to(DEVICE)
        self.VEOD = torch.tensor(3.0).to(DEVICE)

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_state()
        # print("states have ")
        next_states = self.getNextState(states, inputs)
        # print("returned next states")
        output = self.getNextOutput(next_states, inputs)
        # print("returned next output")
        return output, next_states

    def getNextOutput(self, X, U):
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = X.split(1, dim=1)
        i = U
        qSMax = (self.qMax * self.qMaxBASE) * self.VolS / self.Vol
        Tbm = Tb - 273.15
        xpS = qpS / qSMax
        xnS = qnS / qSMax
        xpS = xpS.to(torch.float32).to(DEVICE)
        xnS = xnS.to(torch.float32).to(DEVICE)
        VepMLP = self.MLPp(xpS)
        VenMLP = self.MLPn(xnS)

        safe_log_p = torch.clamp((1 - xpS) / xpS, 1e-18, 1e+18).to(DEVICE)
        safe_log_n = torch.clamp((1 - xnS) / xnS, 1e-18, 1e+18).to(DEVICE)

        Vep = self.U0p + self.R * Tb / self.F * torch.log(safe_log_p) + VepMLP
        Ven = self.U0n + self.R * Tb / self.F * torch.log(safe_log_n) + VenMLP
        Volt = Vep - Ven - Vo - Vsn - Vsp

        # print("Output has size ",Volt.shape)

        return Volt

    def getNextState(self, X, U):
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = X.split(1, dim=1)
        i = U

        # print(Tb.shape,Vo.shape,Vsn.shape,Vsp.shape,qnB.shape,qnS.shape,qpB.shape,qpS.shape)

        qSMax = (self.qMax * self.qMaxBASE) * self.VolS / self.Vol
        xpS = torch.clamp(qpS / qSMax, 1e-18, 1.0)
        xnS = torch.clamp(qnS / qSMax, 1e-18, 1.0)
        Jn0 = 1e-18 + self.kn * (1 - xnS) ** self.alpha * (xnS) ** self.alpha
        # Jn0 = Jn0
        
        Jp0 = 1e-18 + self.kp * (1 - xpS) ** self.alpha * (xpS) ** self.alpha
        # Jp0 = Jp0

        Tbdot = torch.zeros_like(i)
        
        CnBulk = qnB / self.VolB
        CnSurface = qnS / self.VolS
        CpSurface = qpS / self.VolS
        CpBulk = qpB / self.VolB
        qdotDiffusionBSn = (CnBulk - CnSurface) / self.tDiffusion

        qdotDiffusionBSn = qdotDiffusionBSn

        qnBdot = -qdotDiffusionBSn * torch.ones_like(i) 

        qdotDiffusionBSp = (CpBulk - CpSurface) / self.tDiffusion
        # qdotDiffusionBSp = qdotDiffusionBSp

        qpBdot = -qdotDiffusionBSp * torch.ones_like(i) 
        qpSdot = i + qdotDiffusionBSp
        Jn = i / self.Sn
        VoNominal = i * self.Ro * self.RoBASE
        Jp = i / self.Sp
        qnSdot = qdotDiffusionBSn - i

        VsnNominal = self.R * Tb / self.F / self.alpha * torch.asinh(Jn / (2 * Jn0))
        
        Vodot = (VoNominal - Vo) / self.t0
        VspNominal = self.R * Tb / self.F / self.alpha * torch.asinh(Jp / (2 * Jp0))
        Vsndot = (VsnNominal - Vsn) / self.tsn
        Vspdot = (VspNominal - Vsp) / self.tsp
        dt = self.dt

        # Calculate new values for each variable based on dt
        Tb_new = Tb + Tbdot * dt
        Vo_new = Vo + Vodot * dt
        Vsn_new = Vsn + Vsndot * dt
        Vsp_new = Vsp + Vspdot * dt
        qnB_new = qnB + qnBdot * dt
        qnS_new = qnS + qnSdot * dt
        qpB_new = qpB + qpBdot * dt
        qpS_new = qpS + qpSdot * dt
        
        # print(Tb_new.shape,Vo_new.shape,Vsn_new.shape,Vsp_new.shape,qnB_new.shape,qnS_new.shape,qpB_new.shape,qpS_new.shape)

        # Concatenate all the tensors into a single tensor along dimension 1
        XNew = torch.cat([
            Tb_new,  
            Vo_new,
            Vsn_new,
            Vsp_new,
            qnB_new,
            qnS_new,
            qpB_new,
            qpS_new
        ], dim=1)
       
        # print("Update state has size ",XNew.shape)
        return XNew

    def get_initial_state(self):
        self.initBatteryParams(D_trainable=False)
        if self.q_max_model is not None:
            self.qMax = self.q_max_model(torch.tensor(self.curr_cum_pwh).to(DEVICE)) / self.qMaxBASE
        
        if self.R_0_model is not None:
            self.Ro = self.R_0_model(torch.tensor(self.curr_cum_pwh).to(DEVICE)) / self.RoBASE
        
        qpMin = self.qMax * self.qMaxBASE * self.xpMin
        qpSMin = qpMin * self.VolS / self.Vol
        qpBMin = qpMin * self.VolB / self.Vol
        qnMax = self.qMax * self.qMaxBASE * self.xnMax
        qnSMax = qnMax * self.VolS / self.Vol
        qnBMax = qnMax * self.VolB / self.Vol

        if self.initial_state is None:
            initial_state = torch.cat([
                torch.tensor([292.1]).to(DEVICE),
                torch.zeros(3).to(DEVICE),
                qnBMax.reshape(1).to(DEVICE),
                qnSMax.reshape(1).to(DEVICE),
                qpBMin.reshape(1).to(DEVICE),
                qpSMin.reshape(1).to(DEVICE)
            ]).unsqueeze(0).to(DEVICE)
        else:
            initial_state = torch.tensor(self.initial_state).to(DEVICE)
        # print("initial state has size ",initial_state.shape)
        return initial_state

#This is the true RNN cell
class BatteryRNN(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, q_max_base=None, R_0_base=None, D_trainable=False):
        super(BatteryRNN,self).__init__()
        self.q_max_model = q_max_model
        self.R_0_model = R_0_model
        self.curr_cum_pwh = curr_cum_pwh
        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.mlp_trainable = mlp_trainable
        self.q_max_base = q_max_base
        self.R_0_base = R_0_base
        self.D_trainable = D_trainable
        self.cell = BatteryRNNCell(q_max_model=self.q_max_model, R_0_model=self.R_0_model, curr_cum_pwh=self.curr_cum_pwh, initial_state=self.initial_state, dt=self.dt, qMobile=self.qMobile, mlp_trainable=self.mlp_trainable, q_max_base=self.q_max_base, R_0_base=self.R_0_base, D_trainable=self.D_trainable)

    #Define forward pass which is a for loop
    def forward(self, inputs, initial_state=None):
        outputs = []
        if initial_state is None:
            state = self.cell.get_initial_state()

        for t in range(inputs.shape[1]):
            output, state = self.cell(inputs[:,t,:], state)
            outputs.append(output)

        # print(outputs)
        # outputs = torch.stack(outputs)
        # print(outputs.shape)
        return torch.cat(outputs, 1).unsqueeze(-1)