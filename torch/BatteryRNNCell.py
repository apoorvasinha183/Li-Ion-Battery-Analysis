import torch
import torch.nn as nn
import numpy as np

class BatteryRNNCell(nn.Module):
    """This defines the RNN Block for the Battery Cell. This is very primitive as in it only performs 
    """
    def __init__(self, initial_state=None, dt=1.0, qMobile=7600,state_size=8,dtype=torch.float32, **kwargs):
        super(BatteryRNNCell, self).__init__()

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initBatteryParams()

        self.state_size = state_size
        self.output_size = 1
    def build(self, input_shape, **kwargs): # TODO :Why?! What does this do 
        self.built = True

    def initBatteryParams(self):
        """This initializes the constants for the Electrochemical Reactions.
        PSA: These constants define any battery . Please do not create any schemes that update these.
        You are right if you think this is a really weird thing to do."""
        P = self

        P.xnMax = torch.tensor(0.6,device=self.device) # maximum mole fraction (neg electrode)
        P.xnMin = torch.tensor(0,device=self.device)     # minimum mole fraction (neg electrode)
        P.xpMax = torch.tensor(1.0,device=self.device)  # maximum mole fraction (pos electrode)
        P.xpMin = torch.tensor(0.4,device=self.device)  # minimum mole fraction (pos electrode) -> note xn+xp=1
        P.qMax = P.qMobile / (P.xnMax - P.xnMin)    # note qMax = qn+qp
        P.qMax.to(self.device)
        P.Ro = torch.tensor(0.117215,device=self.device) # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)

        # Constants of nature
        P.R = torch.tensor(8.3144621,device=self.device) # universal gas constant, J/K/mol
        P.F = torch.tensor(96487,device=self.device) # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = torch.tensor(0.5,device=self.device)  # anodic/cathodic electrochemical transfer coefficient
        P.Sn = torch.tensor(0.000437545,device=self.device) # surface area (- electrode)
        P.Sp = torch.tensor(0.00030962,device=self.device)  # surface area (+ electrode)
        P.kn = torch.tensor(2120.96,device=self.device) # lumped constant for BV (- electrode)
        P.kp = torch.tensor(248898,device=self.device)  # lumped constant for BV (+ electrode)
        P.Vol = torch.tensor(2e-5,device=self.device) # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = torch.tensor(0.1,device=self.device) # fraction of total volume occupied by surface volume
        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        P.VolS = P.VolSFraction * P.Vol # surface volume
        P.VolS.to(self.device)
        P.VolB = P.Vol - P.VolS # bulk volume
        P.VolB.to(self.device)
        # set up charges (Li ions)
        P.qpMin = P.qMax * P.xpMin    # min charge at pos electrode
        P.qpMin.to(self.device)
        P.qpMax = P.qMax * P.xpMax  # max charge at pos electrode
        P.qpMax.to(self.device)
        P.qpSMin = P.qpMin * P.VolS / P.Vol # min charge at surface, pos electrode
        P.qpSMin.to(self.device)
        P.qpBMin = P.qpMin * P.VolB / P.Vol # min charge at bulk, pos electrode
        P.qpBMin.to(self.device)
        P.qpSMax = P.qpMax * P.VolS / P.Vol # max charge at surface, pos electrode
        P.qpSMax.to(self.device)
        P.qpBMax = P.qpMax * P.VolB / P.Vol # max charge at bulk, pos electrode
        P.qpBMax.to(self.device)
        P.qnMin = P.qMax * P.xnMin # min charge at neg electrode
        P.qnMin.to(self.device)
        P.qnMax = P.qMax * P.xnMax # max charge at neg electrode
        P.qnMax.to(self.device)
        P.qnSMax = P.qnMax * P.VolS / P.Vol # max charge at surface, neg electrode
        P.qnSMax.to(self.device)
        P.qnBMax = P.qnMax * P.VolB / P.Vol # max charge at bulk, neg electrode
        P.qnBMax.to(self.device)
        P.qnSMin = P.qnMin * P.VolS / P.Vol # min charge at surface, neg electrode
        P.qnSMin.to(self.device)
        P.qnBMin = P.qnMin * P.VolB / P.Vol # min charge at bulk, neg electrode
        P.qnBMin.to(self.device)
        P.qSMax = P.qMax * P.VolS / P.Vol # max charge at surface (pos and neg)
        P.qSMax.to(self.device)
        P.qBMax = P.qMax * P.VolB / P.Vol # max charge at bulk (pos and neg)
        P.qBMax.to(self.device)
        # time constants
        P.tDiffusion = torch.tensor(7e6,device=self.device) # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.to = torch.tensor(6.08671,device=self.device) # for Ohmic voltage
        P.tsn = torch.tensor(1001.38,device=self.device) # for surface overpotential (neg)
        P.tsp = torch.tensor(46.4311,device=self.device) # for surface overpotential (pos)
        # Redlich-Kister parameters (positive electrode)
        P.U0p = torch.tensor(4.03,device=self.device)
        P.BASE_Ap0 = torch.tensor(-31593.7,device=self.device)
        P.BASE_Ap1 = torch.tensor(0.106747, device=self.device)
        P.BASE_Ap2 = torch.tensor(24606.4, device=self.device)
        P.BASE_Ap3 = torch.tensor(-78561.9, device=self.device)
        P.BASE_Ap4 = torch.tensor(13317.9, device=self.device)
        P.BASE_Ap5 = torch.tensor(307387.0, device=self.device)
        P.BASE_Ap6 = torch.tensor(84916.1, device=self.device)
        P.BASE_Ap7 = torch.tensor(-1.07469e+06, device=self.device)
        P.BASE_Ap8 = torch.tensor(2285.04, device=self.device)
        P.BASE_Ap9 = torch.tensor(990894.0, device=self.device)
        P.BASE_Ap10 = torch.tensor(283920.0, device=self.device)
        P.BASE_Ap11 = torch.tensor(-161513.0, device=self.device)
        P.BASE_Ap12 = torch.tensor(-469218.0, device=self.device)
       

        P.Ap0 = nn.Parameter(torch.tensor(1.0))
        P.Ap1 = nn.Parameter(torch.tensor(1.0))
        P.Ap2 = nn.Parameter(torch.tensor(1.0))
        P.Ap3 = nn.Parameter(torch.tensor(1.0))
        P.Ap4 = nn.Parameter(torch.tensor(1.0))
        P.Ap5 = nn.Parameter(torch.tensor(1.0))
        P.Ap6 = nn.Parameter(torch.tensor(1.0))
        P.Ap7 = nn.Parameter(torch.tensor(1.0))
        P.Ap8 = nn.Parameter(torch.tensor(1.0))
        P.Ap9 = nn.Parameter(torch.tensor(1.0))
        P.Ap10 = nn.Parameter(torch.tensor(1.0))
        P.Ap11 = nn.Parameter(torch.tensor(1.0))
        P.Ap12 = nn.Parameter(torch.tensor(1.0))
        
        # Redlich-Kister parameters (negative electrode)
        P.U0n = torch.tensor(0.01,device=self.device)
        P.BASE_An0 = torch.tensor(86.19,device=self.device)
        P.An0 = nn.Parameter(torch.tensor(1.0))

        P.An1 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An2 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An3 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An4 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An5 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An6 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An7 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An8 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An9 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An10 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An11 = torch.tensor(0, device=self.device,requires_grad = False)
        P.An12 = torch.tensor(0, device=self.device,requires_grad = False)
        # End of discharge voltage threshold
        P.VEOD = torch.tensor(3.0,device=self.device)

    def forward(self, inputs, states):
        inputs = inputs.to(self.Ap0.device)
        print("States shape is ",states)
        states = states[0,:]

        next_states = self.getNextState(states, inputs)
        output = self.getNextOutput(next_states, inputs)

        return output, [next_states]

    def getAparams(self):
        parameters = self
        Ap0 = parameters.Ap0 * parameters.BASE_Ap0
        Ap1 = parameters.Ap1 * parameters.BASE_Ap1
        Ap2 = parameters.Ap2 * parameters.BASE_Ap2
        Ap3 = parameters.Ap3 * parameters.BASE_Ap3
        Ap4 = parameters.Ap4 * parameters.BASE_Ap4
        Ap5 = parameters.Ap5 * parameters.BASE_Ap5
        Ap6 = parameters.Ap6 * parameters.BASE_Ap6
        Ap7 = parameters.Ap7 * parameters.BASE_Ap7
        Ap8 = parameters.Ap8 * parameters.BASE_Ap8
        Ap9 = parameters.Ap9 * parameters.BASE_Ap9
        Ap10 = parameters.Ap10 * parameters.BASE_Ap10
        Ap11 = parameters.Ap11 * parameters.BASE_Ap11
        Ap12 = parameters.Ap12 * parameters.BASE_Ap12

        An0 = parameters.An0 * parameters.BASE_An0

        return torch.stack([Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,An0])
    def multiply_no_nan(self,x, y):
        # Check for NaN values in both tensors
        nan_mask = torch.isnan(x) | torch.isnan(y)
        
        # Replace NaN values with 1.0 (identity for multiplication)
        x_clean = torch.where(nan_mask, torch.tensor(1.0, dtype=x.dtype, device=x.device), x)
        y_clean = torch.where(nan_mask, torch.tensor(1.0, dtype=y.dtype, device=y.device), y)
        
        # Perform element-wise multiplication
        result = x_clean * y_clean
        
        return result
    def Vi(self, A, x, i):
        # This is a power series approximation of the Vi function
        # I have removed the random commented out stuff and done this normally
        temp_x = 2.0 * x - 1.0
        pow_1 = temp_x ** (i + 1)
        pow_2 = temp_x ** (1 - i)
        temp_2xk = 2.0 * x * i * (1 - x)
        div = temp_2xk / pow_2
        denum = (pow_1 - div) 
        denum = self.multiply_no_nan(denum,A)
        ret = denum / self.F

        return ret
    def safe_Vi(self, A, x, i):
        """I do not know what this does.Looks like a masking meschanism."""
        x_ok = (x != 0.5)
        # Apply the condition using torch.where
        safe_x = torch.where(x_ok, x, torch.ones_like(x))
        
        # Calculate Vi(A, safe_x, i) or return safe_f(x) based on the condition
        result = torch.where(x_ok, self.Vi(A, safe_x, i), torch.zeros_like(x))
        
        return result
    def getNextOutput(self, X, U):
        
        """ OutputEqn   Compute the outputs of the battery model
        #
        #   Z = OutputEqn(parameters,t,X,U,N) computes the outputs of the battery
        #   model given the parameters structure, time, the states, inputs, and
        #   sensor noise. The function is vectorized, so if the function inputs are
        #   matrices, the funciton output will be a matrix, with the rows being the
        #   variables and the columns the samples.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.
        """
        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        P = U[:,0]

        Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,An0 = self.getAparams() # This is really one of the examples why the NASA code is unoptimized
        parameters = self
        Vi = self.Vi

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/parameters.qSMax
        Vep0 = Vi(Ap0,xpS,torch.tensor(0.0, device=self.device))
        Vep1 = Vi(Ap1,xpS,torch.tensor(1.0, device=self.device))
        Vep2 = Vi(Ap2,xpS,torch.tensor(2.0, device=self.device))
        Vep3 = Vi(Ap3,xpS,torch.tensor(3.0, device=self.device))
        Vep4 = Vi(Ap4,xpS,torch.tensor(4.0, device=self.device))
        Vep5 = Vi(Ap5,xpS,torch.tensor(5.0, device=self.device))
        Vep6 = Vi(Ap6,xpS,torch.tensor(6.0, device=self.device))
        Vep7 = Vi(Ap7,xpS,torch.tensor(7.0, device=self.device))
        Vep8 = Vi(Ap8,xpS,torch.tensor(8.0, device=self.device))
        Vep9 = Vi(Ap9,xpS,torch.tensor(9.0, device=self.device))
        Vep10 = Vi(Ap10,xpS,torch.tensor(10.0, device=self.device))
        Vep11 = Vi(Ap11,xpS,torch.tensor(11.0, device=self.device))
        Vep12 = Vi(Ap12,xpS,torch.tensor(12.0, device=self.device))

        xnS = qnS/parameters.qSMax

        Ven0 = Vi(An0,xnS,torch.tensor(0.0, device=self.device))
        Ven1 = Vi(parameters.An1,xnS,torch.tensor(1.0, device=self.device))
        Ven2 = Vi(parameters.An2,xnS,torch.tensor(2.0, device=self.device))
        Ven3 = Vi(parameters.An3,xnS,torch.tensor(3.0, device=self.device))
        Ven4 = Vi(parameters.An4,xnS,torch.tensor(4.0, device=self.device))
        Ven5 = Vi(parameters.An5,xnS,torch.tensor(5.0, device=self.device))
        Ven6 = Vi(parameters.An6,xnS,torch.tensor(6.0, device=self.device))
        Ven7 = Vi(parameters.An7,xnS,torch.tensor(7.0, device=self.device))
        Ven8 = Vi(parameters.An8,xnS,torch.tensor(8.0, device=self.device))
        Ven9 = Vi(parameters.An9,xnS,torch.tensor(9.0, device=self.device))
        Ven10 = Vi(parameters.An10,xnS,torch.tensor(10.0, device=self.device))
        Ven11 = Vi(parameters.An11,xnS,torch.tensor(11.0, device=self.device))
        Ven12 = Vi(parameters.An12,xnS,torch.tensor(12.0, device=self.device))

        Vep = parameters.U0p + parameters.R*Tb/parameters.F*torch.log((1-xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*torch.log((1-xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        V = Vep - Ven - Vo - Vsn - Vsp
        

        return V

    def getNextState(self, X, U):
        """
        # StateEqn   Compute the new states of the battery model
        #
        #   XNew = StateEqn(parameters,t,X,U,N,dt) computes the new states of the
        #   battery model given the parameters strcucture, the current time, the
        #   current states, inputs, process noise, and the sampling time.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.
        """
        """
        This is the Euler integrator. @Antonio : You can think about replacing this with RK4."""
        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        i= U[:,0]
        parameters = self
        xpS = qpS/parameters.qSMax
        xnS = qnS/parameters.qSMax
        Tbdot = torch.zeros(X.shape[0], dtype=self.dtype,device=self.device)
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB
        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn
        Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha
        qpBdot = - qdotDiffusionBSp
        # i = P/V
        qpSdot = i + qdotDiffusionBSp
        Jn = i/parameters.Sn
        VoNominal = i*parameters.Ro
        Jp = i/parameters.Sp
        qnSdot = qdotDiffusionBSn - i
        VsnNominal = parameters.R*Tb/parameters.F/parameters.alpha*torch.asinh(Jn/(2*Jn0))
        Vodot = (VoNominal-Vo)/parameters.to
        VspNominal = parameters.R*Tb/parameters.F/parameters.alpha*torch.asinh(Jp/(2*Jp0))
        Vsndot = (VsnNominal-Vsn)/parameters.tsn
        Vspdot = (VspNominal-Vsp)/parameters.tsp

        dt = self.dt
        # Update state (Euler Integrator)
        XNew = torch.stack([
            Tb + Tbdot*dt,
            Vo + Vodot*dt,
            Vsn + Vsndot*dt,
            Vsp + Vspdot*dt,
            qnB + qnBdot*dt,
            qnS + qnSdot*dt,
            qpB + qpBdot*dt,
            qpS + qpSdot*dt
        ], axis = 1)

        return XNew
    def get_initial_state(self, inputs=None, batch_size=None):
        P = self

        if self.initial_state is None:
            # Compute the initial state using torch operations
            state_size_eff = [batch_size]+[self.state_size]
            #print("state size is ",state_size_eff)
            #another_tensor = torch.tensor([[292.1, 0.0, 0.0, 0.0, P.qnBMax.item(), P.qnSMax.item(), P.qpBMin.item(), P.qpSMin.item()]], dtype=self.dtype)  # 292.1 K, about 18.95 C
            initial_state = torch.ones(state_size_eff, device = self.device) \
                * torch.tensor([[292.1, 0.0, 0.0, 0.0, P.qnBMax.item(), P.qnSMax.item(), P.qpBMin.item(), P.qpSMin.item()]], dtype=self.dtype)  # 292.1 K, about 18.95 C
        else:
            # Convert the initial state to the specified dtype
            initial_state = torch.tensor(self.initial_state, device = self.device)

        return initial_state


if __name__ == "__main__":
    #What do ?!