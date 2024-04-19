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
        self.initBatteryParams(batch_size, D_trainable)

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
        #print("Success!")
        # Initialize MLPn weights
        X = torch.linspace(0.0, 1.0, 100).unsqueeze(1)
        Y = torch.linspace(-8e-4, 8e-4, 100).unsqueeze(1)
        #This is such a chad move
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
        if not D_trainable:
            self.tDiffusion = torch.tensor(7e6) 
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
        #print("states have ")
        next_states = self.getNextState(states, inputs)
        output = self.getNextOutput(next_states, inputs)
        return output, next_states

    def getNextOutput(self, X, U):
        ##print("X has shape ",X.shape)
        Tb, Vo, Vsn, Vsp, qnB, qnS, qpB, qpS = X.split(1, dim=1)
        i = U
        ##print("Input density to Neural net has shape ",qpS.shape)
        qSMax = (self.qMax * self.qMaxBASE) * self.VolS / self.Vol
        ##print("qSMax has shape ",qSMax.shape)
        Tbm = Tb - 273.15
        xpS = qpS / torch.unsqueeze(qSMax,1)
        xnS = qnS / torch.unsqueeze(qSMax,1)
        ##print(type(xpS))
        xpS = xpS.to(torch.float32)
        xnS = xnS.to(torch.float32)
        ##print("Input to Neural net has shape ",xpS.shape)
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
        i = U.clone()
        ##print("X input state has shape ",X.shape)
        ##print("Input current has shape ",i.shape)
        #Annoying
        Tb = torch.squeeze(Tb, dim=1)
        Vo = torch.squeeze(Vo, dim=1)
        Vsn = torch.squeeze(Vsn, dim=1)
        Vsp = torch.squeeze(Vsp, dim=1)
        qnB = torch.squeeze(qnB, dim=1)
        qnS = torch.squeeze(qnS, dim=1)
        qpB = torch.squeeze(qpB, dim=1)
        qpS = torch.squeeze(qpS, dim=1)
        i = torch.squeeze(i,dim=1)
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
        DEBUG = False
        #if DEBUG:
            #print("Vo has shape ",Vo.shape)
            #print("Vnominal has shape ",VoNominal.shape)
            #print("Derivative shape Tb ",Tbdot.shape)
            #print("Derivative shape Vo  ",Vodot.shape)
            #print("Derivative shape Vsp  ",Vspdot.shape)
        #Maturity is realising braodcasting hurts
        
        
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

        # Concatenate all the tensors into a single tensor along dimension 1
        XNew = torch.cat([
            Tb_new.unsqueeze(1),  
            Vo_new.unsqueeze(1),
            Vsn_new.unsqueeze(1),
            Vsp_new.unsqueeze(1),
            qnB_new.unsqueeze(1),
            qnS_new.unsqueeze(1),
            qpB_new.unsqueeze(1),
            qpS_new.unsqueeze(1)
        ], dim=1)
       
        ##print("Update state has size ",XNew.shape)
        return XNew

    def get_initial_state(self, batch_size=None):
        self.initBatteryParams(batch_size, D_trainable=False)
        if self.q_max_model is not None:
            self.qMax = torch.cat([self.q_max_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.qMaxBASE for _ in range(batch_size)], dim=0)
        

        if self.R_0_model is not None:
            self.Ro = torch.cat([self.R_0_model(torch.tensor([[self.curr_cum_pwh]]))[:, 0, 0] / self.RoBASE for _ in range(batch_size)], dim=0)
        
        qpMin = self.qMax * self.qMaxBASE * self.xpMin
        qpSMin = qpMin * self.VolS / self.Vol
        qpBMin = qpMin * self.VolB / self.Vol
        qnMax = self.qMax * self.qMaxBASE * self.xnMax
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
        ##print("initial state has size ",initial_state.shape)
        return initial_state
#This is the true RNN cell
class BatteryRNN(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, batch_size=1, q_max_base=None, R_0_base=None, D_trainable=False):
        super(BatteryRNN,self).__init__()
        self.q_max_model = q_max_model
        self.R_0_model = R_0_model
        self.curr_cum_pwh = curr_cum_pwh
        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.mlp_trainable = mlp_trainable
        self.batch_size = batch_size
        self.q_max_base = q_max_base
        self.R_0_base = R_0_base
        self.D_trainable = D_trainable
        self.cell = BatteryRNNCell(q_max_model=self.q_max_model, R_0_model=self.R_0_model, curr_cum_pwh=self.curr_cum_pwh, initial_state=self.initial_state, dt=self.dt, qMobile=self.qMobile, mlp_trainable=self.mlp_trainable, batch_size=self.batch_size, q_max_base=self.q_max_base, R_0_base=self.R_0_base, D_trainable=self.D_trainable)
    #Define forward pass which is a for loop
    def forward(self, inputs, initial_state=None):
        outputs = []
        if initial_state is None:
            state = self.cell.get_initial_state(batch_size=self.batch_size)
        #print("input has size ",inputs.shape)
        #inputs = inputs[0]
        seq_length = inputs.shape[1]
        #print("sequence length is ",seq_length)
        #print("input state has size ",state.shape)
        outputs = torch.zeros((self.batch_size,seq_length))
        #print("outputs tensor shape is ",outputs.shape)
        for t in range(seq_length):
            input = inputs[:,t,:]
            #print("Input to BIG has shape ",input.shape)
            output, state = self.cell(input, state)
            #print("single step out shape ",output.shape)
            outputs[:,t] = output.squeeze(1)
        ##print("This output is ",torch.shape(outputs))
        ##print("shape of outputs ",np.shape(outputs))
        return outputs