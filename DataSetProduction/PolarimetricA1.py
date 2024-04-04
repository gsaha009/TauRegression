import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import matplotlib.pyplot as plt

from TComplex import TComplex
from util import *


class PolarimetricA1:
    def __init__(self,
                 p4_tau: ak.Array, 
                 p4_os_pi: ak.Array, 
                 p4_ss1_pi: ak.Array, 
                 p4_ss2_pi: ak.Array,
                 p4_a1: ak.Array, # new
                 p4_nu: ak.Array, # new
                 taucharge: int):
        """
            Calculate Polarimetric vector for tau to a1 decay
            All the vectors in the arguments must be with respect to the laboratory frame
            taucharge = +-1
        """
        print("Configure PolarimetricA1 : Start")
        self.p4_tau    = p4_tau
        self.p4_os_pi  = p4_os_pi
        self.p4_ss1_pi = p4_ss1_pi
        self.p4_ss2_pi = p4_ss2_pi
        self.p4_a1 = p4_a1 # new
        self.p4_nu = p4_nu # new
        
        self.taucharge = taucharge
        
        self.mpi        = 0.13957018 # GeV
        self.mpi0       = 0.1349766  # GeV
        self.mtau           = 1.776  # GeV
        self.coscab         = 0.975
        self.mrho           = 0.773  # GeV
        self.mrhoprime      = 1.370  # GeV
        self.ma1            = 1.251  # GeV
        self.mpiprime       = 1.300  # GeV
        self.Gamma0rho      = 0.145  # GeV
        self.Gamma0rhoprime = 0.510  # GeV
        self.Gamma0a1       = 0.599  # GeV
        self.Gamma0piprime  = 0.3    # GeV
        self.fpi            = 0.093  # GeV
        self.fpiprime       = 0.08   # GeV
        self.gpiprimerhopi  = 5.8    # GeV
        self.grhopipi       = 6.08   # GeV
        self.beta           = -0.145
        self.COEF1          = 2.0*np.sqrt(2.)/3.0
        self.COEF2          = -2.0*np.sqrt(2.)/3.0
        self.COEF3          = 2.0*np.sqrt(2.)/3.0 # C AJW 2/98: Add in the D-wave and I=0 3pi substructure:
        self.SIGN           = -taucharge
        self.doSystematic   = False
        self.systType       = "UP"

        #RFdict = self.Setup(self.p4_tau)
        #self.RFdict = RFdict
        #self.LFdict = LFdict

        #print(f"RFdict: {RFdict}")

        print("Configure PolarimetricA1 : End \n")

        
    def Setup(self, RefFrame: ak.Array):
        #print(f"RefFrame_before_rotation: {RefFrame}")

        # ------------------------- NO ROTATION IS APPLIED FOR THIS MOMENT!!! -------------------------- #
        # ------------------------- CHEAK REST OF THE CODE AND DEBUG !!! ------------------------------- #
        # ------------------------- AT THIS MOMENT, RFDICT = LFDICT ------------------------------------ # 
        
        #RotVector = self.p4_tau.pvec # tau 3-vector
        #RefFrame = self.Rotate(RefFrame, RotVector)

        #print(f"RotVector: {RotVector}")
        #print(f"RefFrame_after_rotation: {RefFrame}")
        
        #_tauAlongZLabFrame = self.p4_tau
        #_tauAlongZLabFrame = self.Rotate(_tauAlongZLabFrame, RotVector)
        
        # Boost to rest frame (RF) of tau
        #RF_p4_tau = self.Rotate(self.p4_tau, RotVector)
        #RF_p4_os_pi  = self.Rotate(self.p4_os_pi, RotVector)
        #RF_p4_ss1_pi = self.Rotate(self.p4_ss1_pi, RotVector)
        #RF_p4_ss2_pi = self.Rotate(self.p4_ss2_pi, RotVector)
        #RF_p4_a1 = RF_os_p4_pi + RF_ss1_p4_pi + RF_ss2_p4_pi
        #RF_p4_nu = RF_p4_tau.subtract(RF_p4_a1)
        #RFQ = RF_p4_a1.mass

        #print(f"RF_p4_tau: {RF_p4_tau}")
        #print(f"RF_p4_os_pi: {RF_p4_os_pi}")
        #print(f"RF_p4_ss1_pi: {RF_p4_ss1_pi}")
        #print(f"RF_p4_ss2_pi: {RF_p4_ss2_pi}")
                
        #RFdict = {"p4_tau":RF_p4_tau, "p4_os_pi": RF_p4_os_pi, "p4_ss1_pi": RF_p4_ss1_pi, "p4_ss2_pi": RF_p4_ss2_pi}
        
        # In lab frame

        #RF_p4_tau = self.p4_tau
        #RF_p4_os_pi  = self.p4_os_pi
        #RF_p4_ss1_pi = self.p4_ss1_pi
        #RF_p4_ss2_pi = self.p4_ss2_pi
        # ----------------------- CHECK ---------------------------- #
        # Boost in the tau rest frame, even for tau itself
        # ---------------------------------------------------------- #
        #RF_p4_tau    = self.p4_tau.boost(self.p4_tau.boostvec.negative())
        #RF_p4_tau    = self.p4_tau
        #RF_p4_os_pi  = self.p4_os_pi
        #RF_p4_ss1_pi = self.p4_ss1_pi
        #RF_p4_ss2_pi = self.p4_ss1_pi
       
        #LF_p4_a1 = LF_p4_os_pi + LF_p4_ss1_pi + LF_p4_ss2_pi
        #LFQ = LF_p4_a1.mass
        """
        print(f">> RF_p4_tau: {self.p4_tau}")
        printinfo(self.p4_tau)
        printinfo(self.p4_tau, "pt")
        print(f">> RF_p4_os_pi: {self.p4_os_pi}")
        printinfo(self.p4_os_pi)
        printinfo(self.p4_os_pi, "pt")
        print(f">> RF_p4_ss1_pi: {self.p4_ss1_pi}")
        printinfo(self.p4_ss1_pi)
        printinfo(self.p4_ss1_pi, "pt")
        print(f">> RF_p4_ss2_pi: {self.p4_ss2_pi}")        
        printinfo(self.p4_ss2_pi)
        printinfo(self.p4_ss2_pi, "pt")
        
        # lab frame components are already in constructor
        # still returning due to any probable future possibility
        # for boosting: not sure
        RFdict = {"p4_tau":self.p4_tau, "p4_os_pi": self.p4_os_pi, "p4_ss1_pi": self.p4_ss1_pi, "p4_ss2_pi": self.p4_ss2_pi}

        #return RFdict, LFdict
        print(">> Setup : End \n")
        return RFdict
        """
        pass
    
    def GetVecForPVC(self, a, b, c):
        x = b.x - c.x - a.x*a.x*(b.x - c.x)*(1/a.mass2)
        y = b.y - c.y - a.y*a.y*(b.y - c.y)*(1/a.mass2)
        z = b.z - c.z - a.z*a.z*(b.z - c.z)*(1/a.mass2)
        t = b.t - c.t - a.t*a.t*(b.t - c.t)*(1/a.mass2)
        return ak.zip(
            {
                "x": x,
                "y": y,
                "z": z,
                "t": t,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        

    def PVC(self):
        print(f">> PVC : Start")
        
        #P  = self.RFdict["p4_tau"]
        P  = self.p4_tau
        #print("Tau in Higgs rest frame ===>")
        #printinfo(P)
        
        #q1 = self.RFdict["p4_ss1_pi"]
        q1 = self.p4_ss1_pi
        #print("ss1 Pion ===>")
        #printinfo(q1)

        #q2 = self.RFdict["p4_ss2_pi"]
        q2 = self.p4_ss2_pi
        #print("ss2 Pion ===>")
        #printinfo(q2, 'pt')

        #q3 = self.RFdict["p4_os_pi"]
        q3 = self.p4_os_pi
        #print("os Pion ===>")
        #printinfo(q3)

        #a1 = q1.add(q2.add(q3))
        #a1 = self.p4_a1
        a1 = q1+q2+q3
        print(">> a1 ===>")
        printinfo(a1)
        printinfo(a1, "pt")

        N = P.subtract(a1)
        #N = self.p4_nu
        print(">> Nu ===>")

        N = ak.zip(
            {
                "pt": N.pt,
                "eta": N.eta,
                "phi": N.phi,
                "mass": ak.zeros_like(N.pt)
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior
        )

        printinfo(N)
        printinfo(N, "pt")

        
        s1 = (q2+q3).mass2
        s2 = (q1+q3).mass2
        s3 = (q1+q2).mass2

        #print(q2.x - q3.x - a1.x * (q2.x - q3.x)*(1/a1.mass2) )
        #print(q2 - q3 - a1 * (q2 - q3)*(1/a1.mass2) )

        #print(f"P: {P}")
        #print(f"q1: {q1}")
        #print(f"q2: {q2}")
        #print(f"q3: {q3}")
        #print(f"a1: {a1}")
        #print(f"N: {N}")

        print(f">> s1: ss2_pi + os_pi : {s1}")
        print(f">> s2: ss1_pi + os_pi : {s2}")
        print(f">> s3: ss1_pi + ss2_pi: {s3}")

        plotit(arrlist=[ak.ravel(s1).to_numpy(),
                        ak.ravel(s2).to_numpy(),
                        ak.ravel(s3).to_numpy()])
        
        # Three Lorentzvector: Why??? : No idea!!!
        getvec = lambda a,b,c: b - c - a*(a.dot((b - c))*(1/a.mass2))
        #vec1 = self.GetVecForPVC(a1, q2, q3) # getvec(a1, q2, q3)
        #vec2 = self.GetVecForPVC(a1, q3, q1) # getvec(a1, q3, q1)
        #vec3 = self.GetVecForPVC(a1, q1, q2) # getvec(a1, q1, q2)

        vec1 = getvec(a1, q2, q3)
        vec2 = getvec(a1, q3, q1)
        vec3 = getvec(a1, q1, q2)
        
        print(">> vec1 ===>")
        printinfo(vec1)
        printinfo(vec1, "pt")
        print(">> vec2 ===> ")
        printinfo(vec2)
        printinfo(vec2, "pt")
        print(">> vec3 ===>")
        printinfo(vec3)
        printinfo(vec3, "pt")
        
        #print(f"Complexes: {TComplex(self.COEF1)}\n\t{TComplex(self.COEF2)}\n\t{TComplex(self.COEF3)}")

        print(">> F1 ===>")
        F1 = TComplex(self.COEF1)*self.F3PI(1, a1.mass2, s1, s2)
        print(f">> F1: {F1.Re()}\t{F1.Im()}\n")
        plotit(arrlist=[ak.ravel(F1.Re()).to_numpy(),
                        ak.ravel(F1.Im()).to_numpy()])

        print(">> F2 ===>")
        F2 = TComplex(self.COEF2)*self.F3PI(2, a1.mass2, s2, s1)
        print(f">> F2: {F2.Re()}\t{F2.Im()}\n")
        plotit(arrlist=[ak.ravel(F2.Re()).to_numpy(),
                        ak.ravel(F2.Im()).to_numpy()])
        
        print(">> F3 ===>")
        F3 = TComplex(self.COEF3)*self.F3PI(3, a1.mass2, s3, s1)
        print(f">> F3: {F3.Re()}\t{F3.Im()}\n")
        plotit(arrlist=[ak.ravel(F3.Re()).to_numpy(),
                        ak.ravel(F3.Im()).to_numpy()])

        #var =  TComplex(vec1.energy)*F1 + TComplex(vec2.energy)*F2 + TComplex(vec3.energy)*F3
        #print(f"sdbcjhsdnckjnadkjsnckjsdnckjn: {var.Re()}\t{var.Im()}")
        HADCUR = []

        HADCUR.append(TComplex(vec1.energy)*F1 + TComplex(vec2.energy)*F2 + TComplex(vec3.energy)*F3)
        HADCUR.append(TComplex(vec1.px)*F1 + TComplex(vec2.px)*F2 + TComplex(vec3.px)*F3)
        HADCUR.append(TComplex(vec1.py)*F1 + TComplex(vec2.py)*F2 + TComplex(vec3.py)*F3)
        HADCUR.append(TComplex(vec1.pz)*F1 + TComplex(vec2.pz)*F2 + TComplex(vec3.pz)*F3)
        #HADCUR = {
        #    "HC_E": TComplex(vec1.energy)*F1 + TComplex(vec2.energy)*F2 + TComplex(vec3.energy)*F3,
        #    "HC_Px": TComplex(vec1.px)*F1 + TComplex(vec2.px)*F2 + TComplex(vec3.px)*F3,
        #    "HC_Py": TComplex(vec1.py)*F1 + TComplex(vec2.py)*F2 + TComplex(vec3.py)*F3,
        #    "HC_Pz": TComplex(vec1.pz)*F1 + TComplex(vec2.pz)*F2 + TComplex(vec3.pz)*F3
        #}
        """
        HADCUR = {
            "HC_E": TComplex(vec1.t)*F1 + TComplex(vec2.t)*F2 + TComplex(vec3.t)*F3,
            "HC_Px": TComplex(vec1.x)*F1 + TComplex(vec2.x)*F2 + TComplex(vec3.x)*F3,
            "HC_Py": TComplex(vec1.y)*F1 + TComplex(vec2.y)*F2 + TComplex(vec3.y)*F3,
            "HC_Pz": TComplex(vec1.z)*F1 + TComplex(vec2.z)*F2 + TComplex(vec3.z)*F3
        }
        
        print(f">> HC_E: \n  Re: {HADCUR['HC_E'].Re()}\n  Im: {HADCUR['HC_E'].Im()}")
        plotit(arrlist=[ak.ravel(HADCUR['HC_E'].Re()).to_numpy(),
                        ak.ravel(HADCUR['HC_E'].Im()).to_numpy()])

        print(f">> HC_Px: \n  Re: {HADCUR['HC_Px'].Re()}\n  Im: {HADCUR['HC_Px'].Im()}")
        plotit(arrlist=[ak.ravel(HADCUR['HC_Px'].Re()).to_numpy(),
                        ak.ravel(HADCUR['HC_Px'].Im()).to_numpy()])

        print(f">> HC_Py: \n  Re: {HADCUR['HC_Py'].Re()}\n  Im: {HADCUR['HC_Py'].Im()}")
        plotit(arrlist=[ak.ravel(HADCUR['HC_Py'].Re()).to_numpy(),
                        ak.ravel(HADCUR['HC_Py'].Im()).to_numpy()])

        print(f">> HC_Pz: \n  Re: {HADCUR['HC_Pz'].Re()}\n  Im: {HADCUR['HC_Pz'].Im()}")
        plotit(arrlist=[ak.ravel(HADCUR['HC_Pz'].Re()).to_numpy(),
                        ak.ravel(HADCUR['HC_Pz'].Im()).to_numpy()])
        """
        #print(f"HADCUR: \nHC_E: {HADCUR['HC_E']} \nHC_Px: {HADCUR['HC_Px']} \nHC_Py: {HADCUR['HC_Py']} \nHC_Pz: {HADCUR['HC_Pz']}")
        
        #HADCURC = {key:val.Conjugate() for key,val in HADCUR.items()}
        HADCURC = [val.Conjugate() for val in HADCUR]
        """
        print(f">> C-HC_E: \n  Re: {HADCURC['HC_E'].Re()}\n  Im: {HADCURC['HC_E'].Im()}")
        plotit(arrlist=[ak.ravel(HADCURC['HC_E'].Re()).to_numpy(),
                        ak.ravel(HADCURC['HC_E'].Im()).to_numpy()])

        print(f">> C-HC_Px: \n  Re: {HADCURC['HC_Px'].Re()}\n  Im: {HADCURC['HC_Px'].Im()}")
        plotit(arrlist=[ak.ravel(HADCURC['HC_Px'].Re()).to_numpy(),
                        ak.ravel(HADCURC['HC_Px'].Im()).to_numpy()])

        print(f">> C-HC_Py: \n  Re: {HADCURC['HC_Py'].Re()}\n  Im: {HADCURC['HC_Py'].Im()}")
        plotit(arrlist=[ak.ravel(HADCURC['HC_Py'].Re()).to_numpy(),
                        ak.ravel(HADCURC['HC_Py'].Im()).to_numpy()])

        print(f">> C-HC_Pz: \n  Re: {HADCURC['HC_Pz'].Re()}\n  Im: {HADCURC['HC_Pz'].Im()}")
        plotit(arrlist=[ak.ravel(HADCURC['HC_Pz'].Re()).to_numpy(),
                        ak.ravel(HADCURC['HC_Pz'].Im()).to_numpy()])
        """
        # calling CLVEC
        # and CLAXI
        # find bug there !!!
        CLV = self.CLVEC(HADCUR, HADCURC, N)
        CLA = self.CLAXI(HADCUR, HADCURC, N)

        print(f">> CLV ===>")
        printinfo(CLV)
        print(f">> CLA ===>")
        printinfo(CLA)

        """
        BWProd1 = self.f3(a1.mass)*self.BreitWigner(np.sqrt(s2),"rho")
        BWProd2 = self.f3(a1.mass)*self.BreitWigner(np.sqrt(s1),"rho")

        print(f"  BWProd1: {BWProd1.Re()}\t{BWProd1.Im()}")
        self.plotit(arrlist=[ak.ravel(BWProd1.Re()).to_numpy(),
                             ak.ravel(BWProd1.Im()).to_numpy()],
                    bins=100,
                    log=False,
                    dim=(1,2))        
        print(f"  BWProd2: {BWProd2.Re()}\t{BWProd2.Im()}")
        self.plotit(arrlist=[ak.ravel(BWProd2.Re()).to_numpy(),
                             ak.ravel(BWProd2.Im()).to_numpy()],
                    bins=100,
                    log=False,
                    dim=(1,2))
        """
        
        pclv = P.dot(CLV)
        print(f">> P.CLV: {pclv}")
        pcla = P.dot(CLA)
        print(f">> P.CLA: {pcla}")

        #omega = pclv - self.SIGN*pcla
        omega = pclv - pcla
        print(f">> omega: {omega}")

        plotit(arrlist=[ak.ravel(pclv).to_numpy(),
                        ak.ravel(pcla).to_numpy(),
                        ak.ravel(omega).to_numpy()])

        print(f">> P.mass: {P.mass}")
        #M_ = ak.ones_like(P.mass) * 1.777 # check
        A = (P.mass)**2
        print(f">> P.mass**2: {A}")
        #A = M_**2 # check
        CLAmCLV = CLA.subtract(CLV)
        print(f">> CLA - CLV : {CLAmCLV}")

        #print(f"  P.dot(CLA) - P.dot(CLV): {P.dot(CLA) - P.dot(CLV)}\t -omega: {-omega}")        
        
        print(f">> P*(P.dot(CLA) - P.dot(CLV)): {P*(P.dot(CLA) - P.dot(CLV))}\t{-P*omega}")
        print(f">> 1/omega/P.mass: {1/omega/P.mass}")

        
        #out = ((P.mass)*(P.mass)*(CLA-CLV) - P*(P.dot(CLA) - P.dot(CLV)))*(1/omega/P.mass)
        out = ((self.mtau**2)*(CLA-CLV) - P*(P.dot(CLA) - P.dot(CLV)))*(1/omega/self.mtau)
        #out = ((P.mass)*(P.mass)*(CLA-CLV) + P*omega)*(P.mass/omega)

        print(f">> PVC out : {out}")
        plotit(arrlist=[ak.ravel(out.mass).to_numpy()])

        print(f">> PVC : End ")
        return out


    def F3PI(self,
             IFORM: float,
             QQ: ak.Array,
             SA: ak.Array,
             SB: ak.Array):
        """
            Calculate the F3PIFactor.
        """
        print(">>>> F3PI --- : --- Start ---")
        
        MRO = 0.7743
        GRO = 0.1491
        MRP = 1.370
        GRP = 0.386
        MF2 = 1.275
        GF2 = 0.185
        MF0 = 1.186
        GF0 = 0.350
        MSG = 0.860
        GSG = 0.880
        MPIZ = self.mpi0
        MPIC = self.mpi

        M1 = 0
        M2 = 0
        M3 = 0

        IDK = 1  # It is 3pi

        if IDK == 1:
            M1 = MPIZ
            M2 = MPIZ
            M3 = MPIC
        elif IDK == 2:
            M1 = MPIC
            M2 = MPIC
            M3 = MPIC

        M1SQ = M1*M1
        M2SQ = M2*M2
        M3SQ = M3*M3
        
        
        # parameter varioation for
        # systematics from https://arxiv.org/pdf/hep-ex/9902022.pdf
        db2, dph2 = 0.094, 0.253
        db3, dph3 = 0.094, 0.104
        db4, dph4 = 0.296, 0.170
        db5, dph5 = 0.167, 0.104
        db6, dph6 = 0.284, 0.036
        db7, dph7 = 0.148, 0.063

        scale = 0.0
        if self.doSystematic:
            if self.systType == "UP":
                scale = 1
            elif self.systType == "DOWN":
                scale = -1
                
        # Breit-Wigner functions with isotropic decay angular distribution
        # Real part must be equal to one, stupid polar implemenation in root
        BT1 = TComplex(1., 0.)
        BT2 = TComplex(0.12  + scale*db2, 0.) * TComplex(1, (0.99   +  scale*dph2)*np.pi, True)
        BT3 = TComplex(0.37  + scale*db3, 0.) * TComplex(1, (-0.15  +  scale*dph3)*np.pi, True)
        BT4 = TComplex(0.87  + scale*db4, 0.) * TComplex(1, (0.53   +  scale*dph4)*np.pi, True)
        BT5 = TComplex(0.71  + scale*db5, 0.) * TComplex(1, (0.56   +  scale*dph5)*np.pi, True)
        BT6 = TComplex(2.10  + scale*db6, 0.) * TComplex(1, (0.23   +  scale*dph6)*np.pi, True)
        BT7 = TComplex(0.77  + scale*db7, 0.) * TComplex(1, (-0.54  +  scale*dph7)*np.pi, True)

        print(f">>>> BT1: {BT1.Re()}\t{BT1.Im()}")
        print(f">>>> BT2: {BT2.Re()}\t{BT2.Im()}")
        print(f">>>> BT3: {BT3.Re()}\t{BT3.Im()}")
        print(f">>>> BT4: {BT4.Re()}\t{BT4.Im()}")
        print(f">>>> BT5: {BT5.Re()}\t{BT5.Im()}")
        print(f">>>> BT6: {BT6.Re()}\t{BT6.Im()}")
        print(f">>>> BT7: {BT7.Re()}\t{BT7.Im()}")

        
        #F3PIFactor = TComplex(0., 0.) # initialize to zero
        #print(f">>>> F3PIFactor: {F3PIFactor.Re()}\t{F3PIFactor.Im()}")
        F3PIFactor = None

        
        if IDK == 2:
            print(f"IDK: {IDK}")
            if IFORM == 1 or IFORM == 2:
                print(f">>>> IFORM: {IFORM}")
                S1 = SA
                S2 = SB 
                S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                print(f">>>> S1: {S1}")
                print(f">>>> S2: {S2}")
                print(f">>>> S3: {S3}")
                
                F134 = -(1 / 3.) * ((S3 - M3SQ) - (S1 - M1SQ))
                F15A = -(1 / 2.) * ((S2 - M2SQ) - (S3 - M3SQ))
                F15B = -(1 / 18.) * (QQ - M2SQ + S2) * (2 * M1SQ + 2 * M3SQ - S2) / S2
                F167 = -(2 / 3.)

                print(f">>>> F134: {F134}")
                print(f">>>> F15A: {F15A}")
                print(f">>>> F15B: {F15B}")
                print(f">>>> F167: {F167}")
    
                # Breit Wigners for all the contributions:
                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF21 = self.BWIGML(S1, MF2, GF2, M2, M3, 2)
                FF22 = self.BWIGML(S2, MF2, GF2, M3, M1, 2)
                FSG2 = self.BWIGML(S2, MSG, GSG, M3, M1, 0)
                FF02 = self.BWIGML(S2, MF0, GF0, M3, M1, 0)

                print(f">>>> FRO1: {FRO1}")
                print(f">>>> FRP1: {FRP1}")
                print(f">>>> FRO2: {FRO2}")
                print(f">>>> FRP2: {FRP2}")
                print(f">>>> FF21: {FF21}")
                print(f">>>> FF22: {FF22}")
                print(f">>>> FSG2: {FSG2}")
                print(f">>>> FF02: {FF02}")
                
                F3PIFactor = BT1*FRO1 \
                           + BT2*FRP1 \
                           + BT3*TComplex(F134, 0.)*FRO2 \
                           + BT4*TComplex(F134, 0.)*FRP2 \
                           - BT5*TComplex(F15A, 0.)*FF21 \
                           - BT5*TComplex(F15B, 0.)*FF22 \
                           - BT6*TComplex(F167, 0.)*FSG2 \
                           - BT7*TComplex(F167, 0.)*FF02

                print(f">>>> F3PIFactor: {F3PIFactor}")
                
            elif IFORM == 3:
                print(f">>>> IFORM: {IFORM}")
                S3 = SA 
                S1 = SB
                S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                print(f">>>> S1: {S1}")
                print(f">>>> S2: {S2}")
                print(f">>>> S3: {S3}")
                
                F34A = (1 / 3) * ((S2 - M2SQ) - (S3 - M3SQ))
                F34B = (1 / 3) * ((S3 - M3SQ) - (S1 - M1SQ))
                F35A = -(1 / 18) * (QQ - M1SQ + S1) * (2 * M2SQ + 2 * M3SQ - S1) / S1
                F35B = (1 / 18) * (QQ - M2SQ + S2) * (2 * M3SQ + 2 * M1SQ - S2) / S2
                F36A = -(2 / 3)
                F36B = (2 / 3)

                print(f">>>> F34A: {F34A}")
                print(f">>>> F34B: {F34B}")
                print(f">>>> F35A: {F35A}")
                print(f">>>> F35B: {F35B}")
                print(f">>>> F36A: {F36A}")
                print(f">>>> F36B: {F36B}")

                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF21 = self.BWIGML(S1, MF2, GF2, M2, M3, 2)
                FF22 = self.BWIGML(S2, MF2, GF2, M3, M1, 2)
                FSG1 = self.BWIGML(S1, MSG, GSG, M2, M3, 0)
                FSG2 = self.BWIGML(S2, MSG, GSG, M3, M1, 0)
                FF01 = self.BWIGML(S1, MF0, GF0, M2, M3, 0)
                FF02 = self.BWIGML(S2, MF0, GF0, M3, M1, 0)

                print(f">>>> FRO1: {FRO1}")
                print(f">>>> FRP1: {FRP1}")
                print(f">>>> FRO2: {FRO2}")
                print(f">>>> FRP2: {FRP2}")
                print(f">>>> FF21: {FF21}")
                print(f">>>> FF22: {FF22}")
                print(f">>>> FSG1: {FSG1}")                
                print(f">>>> FSG2: {FSG2}")
                print(f">>>> FF01: {FF01}")
                print(f">>>> FF02: {FF02}")

                
                F3PIFactor = BT3*(TComplex(F34A, 0.)*FRO1 + TComplex(F34B, 0.)*FRO2) \
                           + BT4*(TComplex(F34A, 0.)*FRP1 + TComplex(F34B, 0.)*FRP2) \
                           - BT5*(TComplex(F35A, 0.)*FF21 + TComplex(F35B, 0.)*FF22) \
                           - BT6*(TComplex(F36A, 0.)*FSG1 + TComplex(F36B, 0.)*FSG2) \
                           - BT7*(TComplex(F36A, 0.)*FF01 + TComplex(F36B, 0.)*FF02)

                print(f">>>> F3PIFactor: {F3PIFactor}")
                
        if IDK == 1:
            print(f">>>> IDK: {IDK}")
            if IFORM == 1 or IFORM == 2:
                print(f">>>> IFORM: {IFORM}")
                S1 = SA 
                S2 = SB
                S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                print(f">>>> S1: {S1}")
                print(f">>>> S2: {S2}")
                print(f">>>> S3: {S3}")
                plotit(arrlist=[ak.ravel(S1).to_numpy(),
                                ak.ravel(S2).to_numpy(),
                                ak.ravel(S3).to_numpy()])
                
                # C it is 2pi0pi-
                # C Lorentz invariants for all the contributions:
                F134 = -(1 / 3.)  * ((S3 - M3SQ) - (S1 - M1SQ))                      # array
                F150 =  (1 / 18.) * (QQ - M3SQ + S3) * (2*M1SQ + 2*M2SQ - S3) / S3   # array
                F167 =  (2 / 3.)                                                     # scalar

                print(f">>>> F134: {F134}")
                print(f">>>> F150: {F150}")
                print(f">>>> F167: {F167}")

                # FR**: all are TComplex
                print(">>>> FRO1 ===>")
                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)

                print(">>>> FRP1 ===>")
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)

                print(">>>> FRO2 ===>")
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                
                print(">>>> FRP2 ===>")
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)

                print(">>>> FF23 ===>")
                FF23 = self.BWIGML(S3, MF2, GF2, M1, M2, 2)

                print(">>>> FSG3 ===>")
                FSG3 = self.BWIGML(S3, MSG, GSG, M1, M2, 0)

                print(">>>> FF03 ===>")
                FF03 = self.BWIGML(S3, MF0, GF0, M1, M2, 0)

                
                F3PIFactor = BT1*FRO1 \
                           + BT2*FRP1 \
                           + BT3*TComplex(F134)*FRO2 \
                           + BT4*TComplex(F134)*FRP2 \
                           + BT5*TComplex(F150)*FF23 \
                           + BT6*TComplex(F167)*FSG3 \
                           + BT7*TComplex(F167)*FF03
                print(f">>>> F3PIFactor: {F3PIFactor.Re()}\t{F3PIFactor.Im()}\n\n")
                
            elif IFORM == 3:
                print(f">>>> IFORM: {IFORM}")

                S3 = SA
                S1 = SB 
                S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                print(f">>>> S1: {S1}")
                print(f">>>> S2: {S2}")
                print(f">>>> S3: {S3}")
                plotit(arrlist=[ak.ravel(S1).to_numpy(),
                                ak.ravel(S2).to_numpy(),
                                ak.ravel(S3).to_numpy()])

                F34A = (1 / 3.) * ((S2 - M2SQ) - (S3 - M3SQ)) # array
                F34B = (1 / 3.) * ((S3 - M3SQ) - (S1 - M1SQ)) # array
                F35 = -(1 / 2.) * ((S1 - M1SQ) - (S2 - M2SQ)) # array

                print(f">>>> F34A: {F34A}")
                print(f">>>> F34B: {F34B}")
                print(f">>>> F35: {F35}")
                
                print(">>>> FRO1 ===>")
                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)

                print(">>>> FRP1 ===>")
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)

                print(">>>> FRO2 ===>")
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)

                print(">>>> FRP2 ===>")
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)

                print(">>>> FF23 ===>")
                FF23 = self.BWIGML(S3, MF2, GF2, M1, M2, 2)                    
                
                
                F3PIFactor = BT3*(TComplex(F34A)*FRO1 + TComplex(F34B)*FRO2) \
                           + BT4*(TComplex(F34A)*FRP1 + TComplex(F34B)*FRP2) \
                           + BT5*TComplex(F35)*FF23

                print(f">>>> F3PIFactor: {F3PIFactor.Re()}\t{F3PIFactor.Im()}\n\n")
                
        # plot F3PI factor
        print(">>>> \\\ F3PI ///")
        plotit(arrlist=[ak.ravel(F3PIFactor.Re()).to_numpy(),
                        ak.ravel(F3PIFactor.Im()).to_numpy()])

        FORMA1 = self.FA1A1P(QQ) # TComplex
        print(f">>>> FORMA1: {FORMA1.Re()}\t{FORMA1.Im()}")
        print(">>>> \\\ FORMA1 ///")
        plotit(arrlist=[ak.ravel(FORMA1.Re()).to_numpy(),
                        ak.ravel(FORMA1.Im()).to_numpy()])        
        
        out = F3PIFactor*FORMA1
        print(f">>>> F3PIFactor x FORMA1: {out.Re()}\t{out.Im()}")
        print(">>>> \\\ F3PIFactor x FORMA1 ///")
        plotit(arrlist=[ak.ravel(out.Re()).to_numpy(),
                        ak.ravel(out.Im()).to_numpy()])
        
        print(">>>> F3PI : End\n\n")
        return  out



    # ------- L-wave BreightWigner for rho
    # Breit-Wigner function with isotropic decay angular distribution
    def BWIGML(self,
               S: ak.Array,
               M: float,
               G: float,
               m1: float,
               m2: float,
               L: int):
        print(f">>>>>> BWIGML : Start ")
        print(f">>>>>> S: {S}")
        print(f">>>>>> M: {M}")
        print(f">>>>>> G: {G}")
        print(f">>>>>> m1: {m1}")
        print(f">>>>>> m2: {m2}")
        MP = (m1 + m2)**2 # scalar
        print(f">>>>>> MP: {MP}")
        MM = (m1 - m2)**2 # scalar
        print(f">>>>>> MM: {MM}")
        MSQ = M**2        # scalar
        print(f">>>>>> MSQ: {MSQ}")
        W = np.sqrt(S)    # array
        print(f">>>>>> W: {W}")
        #WGS = 0.0  
        #QS, QM = 0.0, 0.0
        

        # Prepare a WGS array
        # use ak.where 
        wgs = self.GetWGS(S, MP, MM, MSQ, L, G, W, M)
        dummy = ak.zeros_like(wgs)
        mask = (W > m1+m2)
        print(f">>>>>> W > m1+m2: {mask}")
        WGS = ak.where(mask, wgs, dummy)
        """
        if W > m1 + m2:
            QS = np.sqrt(np.abs((S - MP) * (S - MM))) / W
            QM = np.sqrt(np.abs((MSQ - MP) * (MSQ - MM))) / M
            IPOW = 2 * L + 1
            WGS = G * (MSQ / W) * (QS / QM)**IPOW
        """

        print(f">>>>>> WGS: {WGS}")
        plotit(arrlist=[ak.ravel(WGS).to_numpy()])

        num = TComplex(MSQ, 0)
        den = TComplex((MSQ - S), -WGS)
        print(f">>>>>> num - Re: {num.Re()} - Im: {num.Im()}")
        print(f">>>>>> den - Re: {den.Re()} - Im: {den.Im()}")
        plotit(arrlist=[ak.ravel(den.Re()).to_numpy(),
                        ak.ravel(den.Im()).to_numpy()])
        
        out = num/den
        print(f">>>>>> BWIGML out: {out.Re()}\t{out.Im()}")
        plotit(arrlist=[ak.ravel(out.Re()).to_numpy(),
                        ak.ravel(out.Im()).to_numpy()])

        print(f">>>>>> BWIGML : End ")

        return out


    
    def GetWGS(self, S: ak.Array, MP: float, MM: float, MSQ: float, L: int,
               G: float, W: ak.Array, M: float):
        print(f">>>>>>>> GetWGS : Start")
        print(f">>>>>>>> S: {S}")
        print(f">>>>>>>> MP: {MP}")
        print(f">>>>>>>> MM: {MM}")
        print(f">>>>>>>> MSQ: {MSQ}")
        print(f">>>>>>>> L: {L}")
        print(f">>>>>>>> G: {G}")
        print(f">>>>>>>> W: {W}")
        print(f">>>>>>>> M: {M}")
        QS = np.sqrt(np.abs((S - MP) * (S - MM))) / W
        print(f">>>>>>>> QS: {QS}")
        QM = np.sqrt(np.abs((MSQ - MP) * (MSQ - MM))) / M
        print(f">>>>>>>> QM: {QM}")
        IPOW = 2 * L + 1
        print(f">>>>>>>> IPOW: {IPOW}")
        WGS = G * (MSQ / W) * np.power((QS / QM), IPOW)
        print(f">>>>>>>> WGS: {WGS}")
        print(f">>>>>>>> GetWGS : End")
        return WGS

    
    def FA1A1P(self, XMSQ: ak.Array) -> TComplex:
        print(f">>>>>> FA1A1P : Start")
        XM1 = 1.275000
        XG1 = 0.700
        XM2 = 1.461000
        XG2 = 0.250
        BET = TComplex(0.0, 0.0)

        GG1 = XM1*XG1/(1.3281*0.806)
        GG2 = XM2*XG2/(1.3281*0.806)
        XM1SQ = XM1*XM1
        XM2SQ = XM2*XM2

        GF = self.WGA1(XMSQ) # array
        FG1 = GG1*GF
        FG2 = GG2*GF
          
        F1 = TComplex(-XM1SQ)/TComplex(XMSQ - XM1SQ, FG1)
        F2 = TComplex(-XM2SQ)/TComplex(XMSQ - XM2SQ, FG2)
        FA1A1P = F1 + (BET*F2)
        print(f">>>>>> FA1A1P: {FA1A1P}")
        print(f">>>>>> FA1A1P : End")
        return FA1A1P


    def WGA1(self, QQ: ak.Array):
        print(f">>>>>>>> WGA1 : Start")
        # C mass-dependent M*Gamma of a1 through its decays to
        # C.[(rho-pi S-wave) + (rho-pi D-wave) +
        # C.(f2 pi D-wave) + (f0pi S-wave)]
        # C.AND simple K*K S-wave
        MKST = 0.894
        MK   = 0.496
        MK1SQ = (MKST+MK)**2
        MK2SQ = (MKST-MK)**2
        # C coupling constants squared:
        C3PI = 0.2384*0.2384
        CKST = 4.7621*4.7621*C3PI
        # C Parameterization of numerical integral of total width of a1 to 3pi.
        # C From M. Schmidtler, CBX-97-64-Update.

        S = QQ
        print(f">>>>>>>> S: {S}")
        WG3PIC = self.WGA1C(S)
        WG3PIN = self.WGA1N(S)
            
        # C Contribution to M*Gamma(m(3pi)^2) from S-wave K*K, if above threshold
        #GKST = 0.0
        mask = S > MK1SQ
        print(f">>>>>>>> S > MK1SQ: {mask}")
        gskt = np.sqrt((S-MK1SQ)*(S-MK2SQ))/(2.0*S)
        dummy = ak.zeros_like(gskt)
        GKST = ak.where(mask, gskt, dummy)

        print(f">>>>>>>> GKST: {GKST}")
        
        #if S > MK1SQ: 
        #    GKST = np.sqrt((S-MK1SQ)*(S-MK2SQ))/(2.0*S)
        out = C3PI*(WG3PIC+WG3PIN) + (CKST*GKST)
        print(f">>>>>>>> out: {out}")
        print(f">>>>>>>> WGA1 : End")
        return out



    def WGA1C(self, S: ak.Array):
        """
           --------------- Check the ak.where scopes : Essential
        """

        print(f">>>>>>>>>> WGA1C : Start")
        STH = 0.1753 
        Q0  = 5.80900
        Q1  = -3.00980 
        Q2  = 4.57920 
        P0  = -13.91400 
        P1  = 27.67900 
        P2  = -13.39300 
        P3  = 3.19240 
        P4  = -0.10487

        mask1 = S < STH
        mask2 = (S > STH) & (S < 0.823)

        dummy = ak.zeros_like(S)
        ifmask2 = Q0 * ((S - STH)*(S - STH)*(S - STH)) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)*(S - STH))
        elsemask2 = P0 + P1*S + P2*S*S + P3*S*S*S + P4*S*S*S*S
        
        G1_IM = ak.where(
            mask1,
            dummy,
            ak.where(
                mask2, ifmask2, elsemask2
            )
        )
        #Q0 * ((S - STH)**3) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)**2),
        #    P0 + (P1 * S) + (P2 * S * S) + (P3 * S * S * S)  + (P4 * S * S * S * S)
        
        """
        if S < STH:
            G1_IM = 0.0
        elif STH <= S < 0.823:
            G1_IM = Q0 * (S - STH)**3 * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)**2)
        else:
            G1_IM = P0 + P1 * S + P2 * S**2 + P3 * S**3 + P4 * S**4
        """
        print(f">>>>>>>>>> G1_IM: {G1_IM}")
        print(f">>>>>>>>>> WGA1C : End ")
        return G1_IM

    
    def WGA1N(self, S: ak.Array):
        """
           --------------- Check the ak.where scopes : Essential
        """
        
        print(f">>>>>>>>>> WGA1N : Start")
        Q0 = 6.28450
        Q1 = -2.95950
        Q2 = 4.33550
        P0 = -15.41100
        P1 = 32.08800
        P2 = -17.66600
        P3 = 4.93550
        P4 = -0.37498
        STH = 0.1676

        mask1 = S < STH
        mask2 = (S > STH) & (S < 0.823)

        dummy = ak.zeros_like(S)
        #ifmask2 = Q0 * ((S - STH)**3) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)**2)
        ifmask2 = Q0 * ((S - STH)*(S - STH)*(S - STH)) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)*(S - STH))
        #elsemask2 = P0 + (P1 * S) + (P2 * S**2) + (P3 * S**3) + (P4 * S**4)
        elsemask2 = P0 + P1*S + P2*S*S + P3*S*S*S + P4*S*S*S*S
        
        G1_IM = ak.where(
            mask1,
            dummy,
            ak.where(
                mask2, ifmask2, elsemask2
            )
        )
        """
        if S < STH :
            G1_IM = 0.0
        elif STH <= S < 0.823:
            G1_IM = Q0 * (S - STH)**3 * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)**2)
        else:
            G1_IM = P0 + P1 * S + P2 * S**2 + P3 * S**3 + P4 * S**4
        """
        print(f">>>>>>>>>> G1_IM: {G1_IM}")
        print(f">>>>>>>>>> WGA1N : End")
        return G1_IM

    
    
    def CLVEC(self, H: list, HC: list, N: ak.Array) -> ak.Array:
        """
           ------------------- Check the multiplications of two arrays
        """

        print(f">>>> CLVEC --- : --- Start ---")
        """
        HN  = H["HC_E"]*N.energy - (H["HC_Px"]*N.px + H["HC_Py"]*N.py + H["HC_Pz"]*N.pz)
        print(f"  HN: {HN}")
        HCN = HC["HC_E"]*N.energy - (HC["HC_Px"]*N.px + HC["HC_Py"]*N.py + HC["HC_Pz"]*N.pz)
        print(f"  HCN: {HCN}")
        HH  = (H["HC_E"]*HC["HC_E"] - (H["HC_Px"]*HC["HC_Px"] + H["HC_Py"]*HC["HC_Py"] + H["HC_Pz"]*HC["HC_Pz"])).Re()
        print(f"  HH: {HH}")
        
        PIVEC0 = 2*( 2*(HN*HC["HC_E"]).Re()  - HH*N.energy)
        PIVEC1 = 2*( 2*(HN*HC["HC_Px"]).Re() - HH*N.px    )
        PIVEC2 = 2*( 2*(HN*HC["HC_Py"]).Re() - HH*N.py    )
        PIVEC3 = 2*( 2*(HN*HC["HC_Pz"]).Re() - HH*N.pz    )
        """
        #print(f"adslkcnkjdsa kjfnkjewmflkrekjvnlkfdsmvc: {H['HC_E'].Re()}\t{H['HC_E'].Im()}")
        #HN  = H["HC_E"]*N.energy - H["HC_Px"]*N.px - H["HC_Py"]*N.py - H["HC_Pz"]*N.pz # TComplex
        HN  = H[0]*N.energy - H[1]*N.px - H[2]*N.py - H[3]*N.pz # TComplex
        print(f">>>> HN: {HN.Re()}\t{HN.Im()}")
        plotit(arrlist=[ak.ravel(HN.Re()).to_numpy(),
                        ak.ravel(HN.Im()).to_numpy()],
               dim=(1,2), log=True)
        
        #HCN = HC["HC_E"]*N.energy - HC["HC_Px"]*N.px - HC["HC_Py"]*N.py - HC["HC_Pz"]*N.pz # TComplex
        HCN = HC[0]*N.energy - HC[1]*N.px - HC[2]*N.py - HC[3]*N.pz # TComplex
        print(f">>>> HCN: {HCN.Re()}\t{HCN.Im()}")
        plotit(arrlist=[ak.ravel(HCN.Re()).to_numpy(),
                        ak.ravel(HCN.Im()).to_numpy()],
               dim=(1,2), log=True)
        
        #HH  = (H["HC_E"]*HC["HC_E"] - H["HC_Px"]*HC["HC_Px"] - H["HC_Py"]*HC["HC_Py"] - H["HC_Pz"]*HC["HC_Pz"]).Re() # ak.Array
        HH  = (H[0]*HC[0] - H[1]*HC[1] - H[2]*HC[2] - H[3]*HC[3]).Re() # ak.Array
        print(f">>>> HH: {HH}")
        plotit(arrlist=[ak.ravel(HH).to_numpy()], log=True)

        """
        PIVEC0 = 2*( 2*(HN*HC["HC_E"]).Re()  - HH*N.energy)
        PIVEC1 = 2*( 2*(HN*HC["HC_Px"]).Re() - HH*N.px    )
        PIVEC2 = 2*( 2*(HN*HC["HC_Py"]).Re() - HH*N.py    )
        PIVEC3 = 2*( 2*(HN*HC["HC_Pz"]).Re() - HH*N.pz    )
        """

        print(f"HN*HC[0].Re: {(HN*HC[0]).Re()}")
        print(f"HN*HC[1].Re: {(HN*HC[1]).Re()}")
        print(f"HN*HC[2].Re: {(HN*HC[2]).Re()}")
        print(f"HN*HC[3].Re: {(HN*HC[3]).Re()}")

        print(f"HH*N.energy: {HH*N.energy}")
        print(f"HH*N.px: {HH*N.px}")
        print(f"HH*N.py: {HH*N.py}")
        print(f"HH*N.pz: {HH*N.pz}")

        PIVEC0 = 2*( 2*(HN*HC[0]).Re() - HH*N.energy )
        PIVEC1 = 2*( 2*(HN*HC[1]).Re() - HH*N.px     )
        PIVEC2 = 2*( 2*(HN*HC[2]).Re() - HH*N.py     )
        PIVEC3 = 2*( 2*(HN*HC[3]).Re() - HH*N.pz     )

        print(f">>>> PIVEC0: {PIVEC0}")
        print(f">>>> PIVEC1: {PIVEC1}")
        print(f">>>> PIVEC2: {PIVEC2}")
        print(f">>>> PIVEC3: {PIVEC3}")

        out = ak.zip(
            {
                "x": PIVEC1,
                "y": PIVEC2,
                "z": PIVEC3,
                "t": PIVEC0,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )

        print(f">>>> out: {out}")
        print(f">>>> CLVEC : End")
        return out
     
    
    def CLAXI(self, H: list, HC: list, N: ak.Array) -> ak.Array:
        """
           ------------------- Check vectors are the same as the previous function
           ------------------- Check simple operations with TComplex class
           ------------------- & compare with np.comples
        """
        print(f">>>> CLAXI : Start")
        #a1 = HC["HC_Px"]
        #a2 = HC["HC_Py"]
        #a3 = HC["HC_Pz"]
        #a4 = HC["HC_E"]
        a1 = HC[1]
        a2 = HC[2]
        a3 = HC[3]
        a4 = HC[0]
        #print(f"  a1: {a1} \n  a2: {a2} \n  a3:{a3} \n  a4: {a4}")

        #b1 = H["HC_Px"]
        #b2 = H["HC_Py"]
        #b3 = H["HC_Pz"]
        #b4 = H["HC_E"]
        b1 = H[1]
        b2 = H[2]
        b3 = H[3]
        b4 = H[0]
        #print(f"  b1: {b1} \n  b2: {b2} \n  b3:{b3} \n  b4: {b4}")

        c1 = N.px
        c2 = N.py
        c3 = N.pz
        c4 = N.energy   
        #print(f"  c1: {c1} \n  c2: {c2} \n  c3:{c3} \n  c4: {c4}")
        
        d34 = (a3*b4 - a4*b3).Im()
        d24 = (a2*b4 - a4*b2).Im()
        d23 = (a2*b3 - a3*b2).Im()
        d14 = (a1*b4 - a4*b1).Im()
        d13 = (a1*b3 - a3*b1).Im()
        d12 = (a1*b2 - a2*b1).Im()

        print(f">>>> d12: {d12} \n>>>> d13: {d13} \n>>>> d14:{d14} \n>>>> d23: {d23} \n>>>> d24: {d24} \n>>>> d34: {d34}")
        
        #PIAX0 = -self.SIGN*2*(-c1.dot(d23) + c2.dot(d13) - c3.dot(d12))
        #PIAX1 = self.SIGN*2*( c2.dot(d34) - c3.dot(d24) + c4.dot(d23))
        #PIAX2 = self.SIGN*2*(-c1.dot(d34) + c3.dot(d14) - c4.dot(d13))
        #PIAX3 = self.SIGN*2*( c1.dot(d24) - c2.dot(d14) + c4.dot(d12))
        #PIAX0 = -self.SIGN*2*(c2*d13 - c1*d23 - c3*d12)
        #PIAX1 =  self.SIGN*2*(c2*d34 - c3*d24 + c4*d23)
        #PIAX2 =  self.SIGN*2*(c3*d14 - c1*d34 - c4*d13)
        #PIAX3 =  self.SIGN*2*(c1*d24 - c2*d14 + c4*d12)

        PIAX0 = -self.SIGN*2*(-c1*d23 + c2*d13 - c3*d12)
        PIAX1 = self.SIGN*2*(c2*d34 - c3*d24 + c4*d23)
        PIAX2 = self.SIGN*2*(-c1*d34 + c3*d14 - c4*d13)
        PIAX3 = self.SIGN*2*(c1*d24 - c2*d14 + c4*d12)

        print(f">>>> PIAX0: {PIAX0}")
        print(f">>>> PIAX1: {PIAX1}")
        print(f">>>> PIAX2: {PIAX2}")
        print(f">>>> PIAX3: {PIAX3}")
        
        out = ak.zip(
            {
                "x": PIAX1,
                "y": PIAX2,
                "z": PIAX3,
                "t": PIAX0,
            },
            with_name="LorentzVector",
            behavior=vector.behavior,
        )
        print(f">>>> out: {out}")
        print(f">>>> CLAXI : End")
        return out




    """
    
    def FPIKM(self, W, XM1, XM2):
        print(f" --- FPIKM --- : --- Start ---")
        ROM   =  0.773
        ROG   =  0.145
        ROM1  =  1.370
        ROG1  =  0.510
        BETA1 = -0.145

        S = W*W
        print(f"S: {S}")
        L = 1 # P-wave
        out = (self.BWIGML(S,ROM,ROG,XM1,XM2,L) + BETA1 * self.BWIGML(S,ROM1,ROG1,XM1,XM2,L))/(1 + BETA1)
        print(f"out: {out}")
        print(f" --- FPIKM --- : --- End ---")
        return out
    

    def f3(self, Q):
        print(" --- f3 --- : --- Start ---")
        print(f"  Q: {Q}")
        X = self.coscab*2*np.sqrt(2)/3/self.fpi
        Y = self.BreitWigner(Q,"a1")
        print(f"  X: {X}")
        print(f"  Y_re: {Y.Re()}")
        print(f"  Y_im: {Y.Im()}")
        out = Y * X
        print(f"  out: \n  Re: {out.Re()}\n  Im: {out.Im()}")
        print(" --- f3 --- : --- End ---")
        return out
        
        
    def BreitWigner(self, Q, part_type):
        print(f" --- BreitWigner --- : --- Start ---")
        QQ = Q*Q
        print(f"  QQ: {QQ}")
        m  = self.Mass(part_type)
        g  = self.Widths(Q, part_type)
        re = (m*m*(m*m - QQ))/((m*m - QQ)**2 + m*m*g*g)
        print(f"  re: {re}")
        im = Q*g/((m*m - QQ)**2 + m*m*g*g)
        print(f"  im: {im}")
        
        out = TComplex(re, im)
        
        plotit(arrlist=[ak.ravel(out.Re()).to_numpy(),
                             ak.ravel(out.Im()).to_numpy()],
                    bins=100,
                    log=False,
                    dim=(1,2))
        
        print(f" --- BreitWigner --- : --- End ---")
        return out

    
    def Widths(self, Q, part_type):
        print(" ----- Widths --- : --- Start ---")
        QQ = Q*Q
        print(f"  QQ: {QQ}")
        Gamma = None
        #Gamma = self.Gamma0rho*self.mrho*((self.ppi(QQ)/self.ppi(self.mrho*self.mrho))**3) / np.sqrt(QQ)
        print(f"  part_type: {part_type}")
        
        if part_type == "rhoprime":
            Gamma = self.Gamma0rhoprime*QQ/self.mrhoprime/self.mrhoprime

        if part_type == "a1":
            Gamma = self.Gamma0a1*self.ga1(Q)/self.ga1(self.ma1)
            #Gamma=Gamma0a1*ma1*ga1(Q)/ga1(ma1)/Q;

        if part_type == "piprime":
            Gamma = self.Gamma0piprime*((np.sqrt(QQ)/self.mpiprime)**5)*(((1-self.mrho*self.mrho/QQ)/(1-self.mrho*self.mrho/self.mpiprime/self.mpiprime))**3)

        else:
            Gamma = self.Gamma0rho * self.mrho * ((self.ppi(QQ)/self.ppi(self.mrho*self.mrho))**3) / np.sqrt(QQ)
            
        print(f"  Gamma: {Gamma}")
        plotit(arrlist=[ak.ravel(Gamma).to_numpy()],
                    bins=100,
                    log=False,
                    dim=(1,1))
        print(" ----- Widths --- : --- End ---")
        return Gamma

    
    def ga1(self, Q):
        print(" ------- ga1 --- : --- Start ---")
        QQ = Q*Q
        print(f"  QQ: {QQ}")
        mask = QQ > (self.mrho + self.mpi)**2
        print(f"  mask: {mask}")
        print(f"  If: {QQ*(1.623 + 10.38/QQ - 9.32/QQ/QQ + 0.65/QQ/QQ/QQ)}")
        print(f"  Else: {4.1*((QQ - 9*self.mpi*self.mpi)**3) * (1 - 3.3*(QQ - 9*self.mpi*self.mpi) + 5.8*(QQ - 9*self.mpi*self.mpi)**2)}")
        out = None
        if isinstance(Q, (int, float)):
            out = QQ*(1.623 + 10.38/QQ - 9.32/QQ/QQ + 0.65/QQ/QQ/QQ) if mask \
                  else 4.1*((QQ - 9*self.mpi*self.mpi)**3) * (1 - 3.3*(QQ - 9*self.mpi*self.mpi) + 5.8*(QQ - 9*self.mpi*self.mpi)**2)
        else:
            out = ak.where(mask,
                           QQ*(1.623 + 10.38/QQ - 9.32/QQ/QQ + 0.65/QQ/QQ/QQ),
                           4.1*((QQ - 9*self.mpi*self.mpi)**3) * (1 - 3.3*(QQ - 9*self.mpi*self.mpi) + 5.8*(QQ - 9*self.mpi*self.mpi)**2))

            plotit(arrlist=[ak.ravel(out).to_numpy()],
                        bins=100,
                        log=False,
                        dim=(1,1))

        print(f"  out: {out}")
        print(" ------- ga1 --- : --- End ---")
        return out

    
    def Mass(self, part_type):
        print(" ----- Mass --- : --- Start ---")
        m = None

        #m = self.mrho
        if part_type == "rhoprime":
            m = self.mrhoprime
        if part_type == "a1":
            m = self.ma1
        if part_type == "piprime":
            m = self.mpiprime
        else:
            m = self.mrho
        print(f"  Mass: {m}")
        print(" ----- Mass --- : --- End ---")
        return m

    
    def ppi(self, QQ):
        print(" --- ppi --- : --- Start ---")
        #if QQ < 4*self.mpi*self.mpi:
        #    raise RuntimeWarning("Can not compute ppi(Q); root square <0 ; return nan")
        out = 0.5*np.sqrt(QQ - 4*self.mpi*self.mpi)
        print(f"  out: {out}")
        if isinstance(out, ak.Array):
            plotit(arrlist=[ak.ravel(out).to_numpy()],
                   bins=100,
                   log=False,
                   dim=(1,1))
        print(" --- ppi --- : --- End ---")
        return out
    """
