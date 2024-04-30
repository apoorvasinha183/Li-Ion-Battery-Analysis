import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BatteryRNNCell_PINN(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, q_max_base=None, R_0_base=None, D_trainable=False, WARM_START = True):
        super(BatteryRNNCell_PINN, self).__init__()

        # Arguments Initialization
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
        self.double()
        self.mlp_trainable = mlp_trainable

        self.initBatteryParams(D_trainable)

        # Define the NN layers for NextOutput 
        self.lin1 = nn.Linear(2+8+1, 34)
        self.lin3 = nn.Linear(34, 17)
        self.lin4 = nn.Linear(17, 1)
        

        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

        # Define the NN layers for NextState
        self.states_lin1 = nn.Linear(11, 22)
        self.states_lin3 = nn.Linear(22, 22)
        self.states_lin4 = nn.Linear(22, 8)

        self.dropout = nn.Dropout(0.5)


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
            self.qMax = nn.Parameter(initial_q_max,requires_grad=not self.mlp_trainable).to(DEVICE)
        else:
            self.qMax = self.q_max_model(torch.tensor(self.curr_cum_pwh)) / self.qMaxBASE
            self.qMax.to(DEVICE)

        if self.R_0_model is None:
            initial_R_0 = torch.tensor(0.15 / self.R_0_base_value)
            self.Ro = nn.Parameter(initial_R_0,requires_grad=not self.mlp_trainable).to(DEVICE)
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


    def forward(self, inputs, prev_out, states=None):

        if states is None:
            states = self.get_initial_state()
        
        next_states = self.getNextState(states, inputs)
        
        output = self.getNextOutput(next_states, inputs, prev_out)
        
        return output, next_states

    def getNextOutput(self, states, i, prev_out):

        other_inputs = torch.tensor([self.qMax, self.Ro]) # 2
        other_inputs = other_inputs.repeat(i.size(0), 1)
        other_inputs = other_inputs.to(torch.float)
        i = i.to(torch.float) # 1

        inputs = torch.cat([i, other_inputs], dim=1) # 8
        X = torch.cat([states, inputs], dim=1)

        X = self.lin1(X)
        X = self.LeakyReLU(X)
        X = self.lin3(X)
        X = self.LeakyReLU(X)
        X = self.lin4(X)
        X = self.ReLU(X) 

        Volt = prev_out - X

        return Volt

    def getNextState(self, states, i):

        # Repeat other_inputs tensor along dimension 0 to match the size of i
        other_inputs = torch.tensor([self.qMax, self.Ro]) # 2
        other_inputs = other_inputs.repeat(i.size(0), 1)

        # Convert other_inputs to Float data type
        other_inputs = other_inputs.to(torch.float)
        i = i.to(torch.float) # 1

        # Concatenate i and repeated other_inputs into a single tensor
        inputs = torch.cat([i, other_inputs], dim=1) 

        
        # Concatenate states and inputs along dimension 1
        X = torch.cat([states, inputs], dim=1) # 8

        X = self.states_lin1(X)
        X = self.LeakyReLU(X)
        X = self.states_lin3(X)
        X = self.LeakyReLU(X)
        X = self.states_lin4(X)
        XNew = self.ReLU(X)

       
        return XNew
    
    def get_initial_state(self):
        ##### PAIN LIVES HERE #####
        #self.initBatteryParams(D_trainable=False)
        #if self.q_max_model is not None:
        #    self.qMax = self.q_max_model(torch.tensor(self.curr_cum_pwh).to(DEVICE)) / self.qMaxBASE
        
        #if self.R_0_model is not None:
        #    self.Ro = self.R_0_model(torch.tensor(self.curr_cum_pwh).to(DEVICE)) / self.RoBASE
        #### PAIN LIVES ABOVE #######
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
class BatteryRNN_PINN(nn.Module):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, q_max_base=None, R_0_base=None, D_trainable=False, WARM_START=True):
        super(BatteryRNN_PINN,self).__init__()
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
        self.WARM_START = WARM_START
        self.cell = BatteryRNNCell_PINN(q_max_model=self.q_max_model, R_0_model=self.R_0_model, curr_cum_pwh=self.curr_cum_pwh, initial_state=self.initial_state, dt=self.dt, qMobile=self.qMobile, mlp_trainable=self.mlp_trainable, q_max_base=self.q_max_base, R_0_base=self.R_0_base, D_trainable=self.D_trainable, WARM_START=True)

    #Define forward pass which is a for loop
    def forward(self, inputs, initial_state=None):
        outputs = []
        if initial_state is None:
            state = self.cell.get_initial_state()
        
        state = state.repeat(inputs.size(0), 1)
        for t in range(inputs.shape[1]):
            if t==0:
                prev_out = torch.full((inputs.size(0), 1), 4.2, device=inputs.device)
            else:
                prev_out = outputs[-1].squeeze(-1)
            output, state = self.cell(inputs = inputs[:,t,:], states= state, prev_out=prev_out)
            outputs.append(output.unsqueeze(-1))

        
        return torch.cat(outputs, 2)