import os
import copy
import numpy as np
import awkward as ak
from PhiCPBase import PhiCPBase
from PolarimetricA1 import PolarimetricA1
#from PolarimetricA1CV import PolarimetricA1
from util import *


class PhiCPComp(PhiCPBase):
    def __init__(self,
                 cat: str = "",
                 taum: ak.Array = None, 
                 taup: ak.Array = None, 
                 taum_decay: ak.Array = None, 
                 taup_decay: ak.Array = None):
        """
            cat: category to compute on
                 possible values would be "pipi", "pirho", "rhorho", "a1rho", "a1a1" [for now]
            Resources:
                 https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.LorentzVector.html#coffea.nanoevents.methods.vector.LorentzVector.boost
        """
        super(PhiCPComp, self).__init__(taum=taum, taup=taup, taum_decay=taum_decay, taup_decay=taup_decay)
        if cat not in ["pipi", "pirho", "rhorho", "a1pi", "a1rho", "a1a1"]:
            raise RuntimeError ("category must be defined !")
        self.cat = cat


    
    # ------------------------------------------------------------------------------- #
    #                                 getvec - pi                                     #
    # ------------------------------------------------------------------------------- #
    def gethvec_pi(self, boostvec, p4_pi):
        print(" --- gethvec_pi --- ")
        print(f"Pi: in lab frame: ")
        printinfo(p4_pi, "p")

        p4_pi_hrest = self.setboost(p4_pi, boostvec)

        print(f"Pi: in rest frame: ")
        printinfo(p4_pi_hrest, "p")
        
        return p4_pi_hrest.pvec

    

        
    # ------------------------------------------------------------------------------- #
    #                                 getvec - rho                                    #
    # ------------------------------------------------------------------------------- #
    def gethvec_rho(self, 
                    boostvec: ak.Array, 
                    p4_tau: ak.Array, 
                    p4_pi: ak.Array, 
                    p4_pi0: ak.Array, 
                    p4_nu: ak.Array):
        print(" --- gethvec_rho --- ")
        print("Lab frame: ")
        print("  Tau: ")
        printinfo(p4_tau, "p")
        print("  Pi: ")
        printinfo(p4_pi, "p")
        print("  Pi0: ")
        printinfo(p4_pi0, "p")
        #print("  Nu: ")
        #printinfo(p4_nu, "p")
        
        
        Tau = self.setboost(p4_tau, boostvec)
        pi  = self.setboost(p4_pi, boostvec)
        pi0 = self.setboost(p4_pi0, boostvec)
        q   = pi.subtract(pi0)
        P   = Tau
        N   = P.subtract(pi.add(pi0))
        #N   = p4_nu.boost(boostvec.negative())

        print("Rest frame: ")
        print("  Tau: ")
        printinfo(Tau)
        print("  Pi: ")
        printinfo(pi)
        print("  Pi0: ")
        printinfo(pi0)
        print("  Pi-Pi0: ")
        printinfo(q)

        out = (((2*(q.dot(N))*q.pvec).subtract(q.mass2*N.pvec)))
        #out = (((2*(q.dot(N))*q.pvec).subtract(self.Mag2(q)*N.pvec)))
        print(f"hvec raw: {out}")
        printinfoP3(out)

        print(f"mag: {out.absolute()}")
        plotit(arrlist=[ak.ravel(out.absolute()).to_numpy()],
               bins=100)
        
        return out, P


    

    # ------------------------------------------------------------------------------- #
    #                                  getvec - a1                                    #
    # ------------------------------------------------------------------------------- #
    def gethvec_a1(self,
                   p4_h: ak.Array,
                   boostvec: ak.Array,
                   p4_tau: ak.Array, 
                   p4_os_pi: ak.Array, 
                   p4_ss1_pi: ak.Array, 
                   p4_ss2_pi: ak.Array,
                   charge: int) -> (ak.Array, ak.Array):
        print(" --- gethvec_a1 --- ")
        print("  ===> tau in lab frame ===>")
        printinfo(p4_tau)
        printinfo(p4_tau, "pt")
        print("  ===> os pion in lab frame ===>")
        printinfo(p4_os_pi)
        printinfo(p4_os_pi, "pt")
        print("  ===> ss pion1 in lab frame ===>")
        printinfo(p4_ss1_pi)
        printinfo(p4_ss1_pi, "pt")
        print("  ===> ss pion2 in lab frame ===>")
        printinfo(p4_ss2_pi)
        printinfo(p4_ss2_pi, "pt")
        print("  ===> a1 in lab frame ===>")
        A = p4_os_pi + p4_ss1_pi + p4_ss2_pi
        printinfo(A)
        printinfo(A, "pt")

        print("  ===> Boostvec: Higgs Rest Frame ===>")
        printinfoP3(boostvec)


        print("  -ve boost applied on tau and its decay products")
        p4_tau_HRF = self.setboost(p4_tau, boostvec)
        p4_os_pi_HRF = self.setboost(p4_os_pi, boostvec)
        p4_ss1_pi_HRF = self.setboost(p4_ss1_pi, boostvec)
        p4_ss2_pi_HRF = self.setboost(p4_ss2_pi, boostvec)

        # check
        a1_temp_p4_HRF = self.setboost(p4_os_pi+p4_ss1_pi+p4_ss2_pi, boostvec)
        nu_temp_p4_HRF = self.setboost(p4_tau - (p4_os_pi+p4_ss1_pi+p4_ss2_pi), boostvec)
        
        s1 = p4_ss2_pi + p4_os_pi
        s2 = p4_ss1_pi + p4_os_pi
        s3 = p4_ss1_pi + p4_ss2_pi
        s1_boost_ind = p4_ss2_pi_HRF + p4_os_pi_HRF
        s2_boost_ind = p4_ss1_pi_HRF + p4_os_pi_HRF
        s3_boost_ind = p4_ss1_pi_HRF + p4_ss2_pi_HRF
        s1_boost_com = self.setboost(s1, boostvec)
        s2_boost_com = self.setboost(s2, boostvec)
        s3_boost_com = self.setboost(s3, boostvec)
        
        print("  ===> tau in Higgs rest frame ===>")
        printinfo(p4_tau_HRF)
        printinfo(p4_tau_HRF, "pt")
        print("  ===> os pion in Higgs rest frame ===>")
        printinfo(p4_os_pi_HRF)
        printinfo(p4_os_pi_HRF, "pt")
        print("  ===> ss pion1 in Higgs rest frame ===>")
        printinfo(p4_ss1_pi_HRF)
        printinfo(p4_ss1_pi_HRF, "pt")
        print("  ===> ss pion2 in Higgs rest frame ===>")
        printinfo(p4_ss2_pi_HRF)
        printinfo(p4_ss2_pi_HRF, "pt")
        print("  ===> temp a1 in Higgs rest frame ===>")
        printinfo(a1_temp_p4_HRF)
        printinfo(a1_temp_p4_HRF, "pt")
        print("  ===> temp nu in Higgs rest frame ===>")
        printinfo(nu_temp_p4_HRF)
        printinfo(nu_temp_p4_HRF, "pt")
        print("  ===> s1,s2,s3 mass in lab frame")
        plotit(arrlist=[ak.ravel(s1.mass2).to_numpy(),
                        ak.ravel(s2.mass2).to_numpy(),
                        ak.ravel(s3.mass2).to_numpy()])
        print("  ===> s1,s2,s3 mass in rest frame: individual boost")
        plotit(arrlist=[ak.ravel(s1_boost_ind.mass2).to_numpy(),
                        ak.ravel(s2_boost_ind.mass2).to_numpy(),
                        ak.ravel(s3_boost_ind.mass2).to_numpy()])
        print("  ===> s1,s2,s3 mass in rest frame: combined boost")
        plotit(arrlist=[ak.ravel(s1_boost_com.mass2).to_numpy(),
                        ak.ravel(s2_boost_com.mass2).to_numpy(),
                        ak.ravel(s3_boost_com.mass2).to_numpy()])

        # -------------------------------------------------------- #
        #                           CV                             #
        # -------------------------------------------------------- #
        """
        boost_ttrf = boostvec
        tauP4_ttrf = self.getP4_rf(p4_tau, boost_ttrf)        
        r, n, k = self.getHelicityAxes(boost_ttrf, tauP4_ttrf, p4_h, p4_tau)
        print("tau in helicity basis")
        tauP4_hf   = self.getP4_hf(tauP4_ttrf, r, n, k)
        boost_trf  = tauP4_hf.boostvec
        print("    boost_trf: ")
        printinfoP3(boost_trf)

        
        # boost on decay products
        print(f" -----> Tau")
        p4_tau_trf = self.getP4_ttrf_hf_trf(p4_tau, boost_ttrf, r, n, k, boost_trf)
        print(f" -----> OS Pion")
        p4_os_pi_trf = self.getP4_ttrf_hf_trf(p4_os_pi, boost_ttrf, r, n, k, boost_trf)
        
        print(f" -----> SS1 Pion")
        p4_ss1_pi_trf = self.getP4_ttrf_hf_trf(p4_ss1_pi, boost_ttrf, r, n, k, boost_trf)

        print(f" -----> SS2 Pion")
        p4_ss2_pi_trf = self.getP4_ttrf_hf_trf(p4_ss2_pi, boost_ttrf, r, n, k, boost_trf)
        # -------------------------------------------------------- #
        #                           CV                             #
        # -------------------------------------------------------- #
        
        print("Christian")
        #a1pol = PolarimetricA1(p4_tau, p4_os_pi_HRF, p4_ss1_pi_HRF, p4_ss2_pi_HRF, charge, "k3ChargedPi")
        a1pol = PolarimetricA1(p4_tau_trf, p4_os_pi_trf, p4_ss1_pi_trf, p4_ss2_pi_trf, charge, "k3ChargedPi")
        out = a1pol.PVC()
        """
        print("Vladimir")
        a1pol  = PolarimetricA1(p4_tau_HRF, p4_os_pi_HRF, p4_ss1_pi_HRF, p4_ss2_pi_HRF,
                                a1_temp_p4_HRF, nu_temp_p4_HRF,
                                charge)
        out = -a1pol.PVC().pvec
        
        print(f"  hvec : {out}")
        printinfoP3(out, plot=True)
        return out, p4_tau_HRF



    # ------------------------------------------------------------------------------- #
    #                                     a1 - a1                                     #
    # ------------------------------------------------------------------------------- #
    def gethvecs_a1a1(self):
        print(" \n--- gethvecs_a1a1 --- ")
        evtdict = self.get_evtinfo_a1a1()

        # Polarimetric vectors will be along the direction of the respective pions
        h1raw, p4_taum_hrest = self.gethvec_a1(evtdict.taum.p4tau + evtdict.taup.p4tau,
                                               evtdict.boostv,
                                               evtdict.taum.p4tau,
                                               evtdict.taum.p4_os_pi,
                                               evtdict.taum.p4_ss1_pi,
                                               evtdict.taum.p4_ss2_pi,
                                               -1)
        h2raw, p4_taup_hrest = self.gethvec_a1(evtdict.taum.p4tau + evtdict.taup.p4tau,
                                               evtdict.boostv,
                                               evtdict.taup.p4tau,
                                               evtdict.taup.p4_os_pi,
                                               evtdict.taup.p4_ss1_pi,
                                               evtdict.taup.p4_ss2_pi,
                                               +1)
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest

    
    
    
    # ------------------------------------------------------------------------------- #
    #                                     pi - pi                                     #
    # ------------------------------------------------------------------------------- #
    def gethvecs_pipi(self):
        evtdict = self.get_evtinfo_pipi()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_pi(evtdict.boostv,
                                evtdict.taum.p4_pi)
        h2raw = self.gethvec_pi(evtdict.boostv,
                                evtdict.taup.p4_pi)

        return h1raw, h2raw, evtdict.taum.p4tau_rest, evtdict.taup.p4tau_rest



    
    # ------------------------------------------------------------------------------- #
    #                                    pi - rho                                     #
    # ------------------------------------------------------------------------------- #
    def gethvecs_pirho_leg1(self):
        evtdict = self.get_evtinfo_pirho_leg1()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw, p4_taum_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taum.p4tau,
                                                evtdict.taum.p4_pi,
                                                evtdict.taum.p4_pi0,
                                                evtdict.taum.p4_nu)
        h2raw = self.gethvec_pi(evtdict.boostv,
                                evtdict.taup.p4_pi)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, evtdict.taup.p4tau_rest

    
    def gethvecs_pirho_leg2(self):
        evtdict = self.get_evtinfo_pirho_leg2()

        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_pi(evtdict.boostv, evtdict.taum.p4_pi)
        h2raw, p4_taup_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taup.p4tau,
                                                evtdict.taup.p4_pi,
                                                evtdict.taup.p4_pi0,
                                                evtdict.taup.p4_nu)
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, evtdict.taum.p4tau_rest, p4_taup_hrest



    
    # ------------------------------------------------------------------------------- #
    #                                    rho - rho                                    #
    # ------------------------------------------------------------------------------- #
    def gethvecs_rhorho(self):
        evtdict = self.get_evtinfo_rhorho()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw, p4_taum_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taum.p4tau,
                                                evtdict.taum.p4_pi,
                                                evtdict.taum.p4_pi0,
                                                evtdict.taum.p4_nu)
        h2raw, p4_taup_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taup.p4tau,
                                                evtdict.taup.p4_pi,
                                                evtdict.taup.p4_pi0,
                                                evtdict.taup.p4_nu)

        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest


    
    
    # ------------------------------------------------------------------------------- #
    #                                     a1 - rho                                    #
    # ------------------------------------------------------------------------------- #
    def gethvecs_a1rho_leg1(self):
        evtdict = self.get_evtinfo_a1rho_leg1()
        h1raw, p4_taum_hrest = self.gethvec_a1(evtdict.taum.p4tau + evtdict.taup.p4tau,
                                               evtdict.boostv,
                                               evtdict.taum.p4tau,
                                               evtdict.taum.p4_os_pi,
                                               evtdict.taum.p4_ss1_pi,
                                               evtdict.taum.p4_ss2_pi,
                                               -1)
        h2raw, p4_taup_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taup.p4tau,
                                                evtdict.taup.p4_pi,
                                                evtdict.taup.p4_pi0,
                                                evtdict.taup.p4_nu)
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest
        
    def gethvecs_a1rho_leg2(self):
        evtdict = self.get_evtinfo_a1rho_leg2()
        h1raw, p4_taum_hrest = self.gethvec_rho(evtdict.boostv,
                                                evtdict.taum.p4tau,
                                                evtdict.taum.p4_pi,
                                                evtdict.taum.p4_pi0,
                                                evtdict.taum.p4_nu)
        h2raw, p4_taup_hrest = self.gethvec_a1(evtdict.taum.p4tau + evtdict.taup.p4tau,
                                               evtdict.boostv,
                                               evtdict.taup.p4tau,
                                               evtdict.taup.p4_os_pi,
                                               evtdict.taup.p4_ss1_pi,
                                               evtdict.taup.p4_ss2_pi,
                                               -1)
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest



    
    # ------------------------------------------------------------------------------- #
    #                                      a1 - pi                                    #
    # ------------------------------------------------------------------------------- #
    def gethvecs_a1pi_leg1(self):
        evtdict = self.get_evtinfo_a1pi_leg1()
        #h_boostvec, p4_taum, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1pi_leg1()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_a1(evtdict.boostv,
                                evtdict.taum.p4tau,
                                evtdict.taum.p4_os_pi,
                                evtdict.taum.p4_ss1_pi,
                                evtdict.taum.p4_ss2_pi,
                                -1)
        h2raw = self.gethvec_pi(evtdict.boostv,
                                evtdict.taup.p4_pi)
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest

    
    def gethvecs_a1pi_leg2(self):
        evtdict = self.get_evtinfo_a1pi_leg2()
        #h_boostvec, p4_taup, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_a1pi_leg2()
        
        # Polarimetric vectors will be along the direction of the respective pions
        h1raw = self.gethvec_a1(evtdict.boostv,
                                evtdict.taup.p4tau,
                                evtdict.taup.p4_os_pi,
                                evtdict.taup.p4_ss1_pi,
                                evtdict.taup.p4_ss2_pi,
                                1)
        h2raw = self.gethvec_pi(evtdict.boostv,
                                evtdict.taum.p4_pi)
        
        #h1raw.absolute(), h2raw.absolute()
        
        return h1raw, h2raw, p4_taum_hrest, p4_taup_hrest


    
    # ------------------------------------------------------------------------------- #
    #                                      hvecs                                      #
    # ------------------------------------------------------------------------------- #
    def gethvecs(self) -> ak.Array:
        print(" --- gethvecs --- ")
        h1 = None
        h2 = None
        p4_taum_hrest = None
        p4_taup_hrest = None
        
        if self.cat == "pipi":
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_pipi()
        elif self.cat == "pirho":
            h1_1, h2_1, p4_taum_hrest_1, p4_taup_hrest_1 = self.gethvecs_pirho_leg1()
            h1_2, h2_2, p4_taum_hrest_2, p4_taup_hrest_2 = self.gethvecs_pirho_leg2()
            h1 = ak.concatenate([h1_1, h1_2], axis=0)
            h2 = ak.concatenate([h2_1, h2_2], axis=0)
            print(p4_taum_hrest_1.pt.type, p4_taum_hrest_2.pt.type)
            p4_taum_hrest = ak.concatenate([p4_taum_hrest_1, p4_taum_hrest_2], axis=0)
            p4_taup_hrest = ak.concatenate([p4_taup_hrest_1, p4_taup_hrest_2], axis=0)
        elif self.cat == "rhorho":
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_rhorho()            
        elif self.cat == "a1a1":
            # create pol_a1 object
            # call configure
            # call gethvec
            h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs_a1a1()
        elif self.cat == "a1pi":
            h1_1, h2_1, p4_taum_hrest_1, p4_taup_hrest_1 = self.gethvecs_a1pi_leg1()
            h1_2, h2_2, p4_taum_hrest_2, p4_taup_hrest_2 = self.gethvecs_a1pi_leg2()
            h1 = ak.concatenate([h1_1, h1_2], axis=0)
            h2 = ak.concatenate([h2_1, h2_2], axis=0)
            print(p4_taum_hrest_1.pt.type, p4_taum_hrest_2.pt.type)
            p4_taum_hrest = ak.concatenate([p4_taum_hrest_1, p4_taum_hrest_2], axis=0)
            p4_taup_hrest = ak.concatenate([p4_taup_hrest_1, p4_taup_hrest_2], axis=0)
        elif self.cat == "a1rho":
            h1_1, h2_1, p4_taum_hrest_1, p4_taup_hrest_1 = self.gethvecs_a1rho_leg1()
            h1_2, h2_2, p4_taum_hrest_2, p4_taup_hrest_2 = self.gethvecs_a1rho_leg2()
            h1 = ak.concatenate([h1_1, h1_2], axis=0)
            h2 = ak.concatenate([h2_1, h2_2], axis=0)
            print(p4_taum_hrest_1.pt.type, p4_taum_hrest_2.pt.type)
            p4_taum_hrest = ak.concatenate([p4_taum_hrest_1, p4_taum_hrest_2], axis=0)
            p4_taup_hrest = ak.concatenate([p4_taup_hrest_1, p4_taup_hrest_2], axis=0)
            
        else:
            raise RuntimeError ("Give right category name")
            
        return h1, h2, p4_taum_hrest, p4_taup_hrest


    

    # ------------------------------------------------------------------------------- #
    #                               Compute PhiCP angle                               #
    # ------------------------------------------------------------------------------- #
    def comp_phiCP(self) -> ak.Array:
        print(" --- comp_phiCP --- ")
        h1, h2, p4_taum_hrest, p4_taup_hrest = self.gethvecs()
        h1 = h1.unit  # new
        h2 = h2.unit  # new
        print("h1: unitvec")
        printinfoP3(h1)
        print("h2: unitvec")
        printinfoP3(h2)

        taum_hrest_pvec_unit = p4_taum_hrest.pvec.unit
        taup_hrest_pvec_unit = p4_taup_hrest.pvec.unit
        print("tau- hrest unit")
        printinfoP3(taum_hrest_pvec_unit)
        print("tau+ hrest unit")
        printinfoP3(taup_hrest_pvec_unit)
        
        #k1raw = h1.cross(p4_taum_hrest.pvec)
        #k2raw = h2.cross(p4_taup_hrest.pvec)
        k1raw = h1.cross(taum_hrest_pvec_unit) # new
        k2raw = h2.cross(taup_hrest_pvec_unit) # new
        #k1raw.absolute(), k2raw.absolute()

        print("k1raw")
        printinfoP3(k1raw, log=True)
        print("k2raw")
        printinfoP3(k2raw, log=True)
        
        k1 = k1raw.unit
        k2 = k2raw.unit

        print("k1: unitvec")
        printinfoP3(k1)
        print("k2: unitvec")
        printinfoP3(k2)

        print(" -- Check -- : -- Strat --")
        print(" -- cosine angle between hi and tau+- ")
        cos_h1_taum = np.arccos(h1.dot(taum_hrest_pvec_unit))
        cos_h1_taup = np.arccos(h1.dot(taup_hrest_pvec_unit))
        cos_h2_taum = np.arccos(h2.dot(taum_hrest_pvec_unit))
        cos_h2_taup = np.arccos(h2.dot(taup_hrest_pvec_unit))
        plotit(arrlist=[(180/np.pi)*ak.ravel(cos_h1_taum).to_numpy(),
                        (180/np.pi)*ak.ravel(cos_h1_taup).to_numpy(),
                        (180/np.pi)*ak.ravel(cos_h2_taum).to_numpy(),
                        (180/np.pi)*ak.ravel(cos_h2_taup).to_numpy()],
               log=True)
        print(" -- h1xh2 --")
        h1xh2 = h1.cross(h2)
        printinfoP3(h1xh2, log=True)
        print("h1xh2.dot(p4_taum_hrest.pvec.unit) : h1xh2.dot(p4_taup_hrest.pvec.unit)")
        angle1 = (180/np.pi)*np.arccos(h1xh2.dot(p4_taum_hrest.pvec.unit))
        angle2 = (180/np.pi)*np.arccos(h1xh2.dot(p4_taup_hrest.pvec.unit))
        plotit(arrlist=[ak.ravel(angle1).to_numpy(),
                        ak.ravel(angle2).to_numpy()],
               log=True)

        print(" -- Check -- : -- End --")
        
        angle = (h1.cross(h2)).dot(p4_taum_hrest.pvec.unit)
        print(f"Angle: {angle}")
        plotit(arrlist=[ak.ravel(angle).to_numpy()], bins=50, log=True)
        
        temp = np.arccos(k1.dot(k2))
        temp2 = 2*np.pi - temp
        #temp = np.arctan2((k1.cross(k2)).absolute(), k1.dot(k2)) # new
        #temp2 = (2*np.pi - temp)                                 # new
        plotit(arrlist=[ak.ravel(temp).to_numpy(),
                        ak.ravel(temp2).to_numpy()],
               bins=9,
               log=False,
               dim=(1,2))
        phicp = ak.where(angle <= 0.0, temp, temp2)

        print(f"PhiCP: {phicp}")
        plotit(arrlist=[ak.ravel(phicp).to_numpy()], bins=9)
        
        return phicp, {"h1_unit": h1, "h2_unit": h2, "k1_unit": k1, "k2_unit": k2, "k1_raw": k1raw, "k2_raw": k2raw}

    

    
    # ------------------------------------------------------------------------------- #
    #                   PhiCP angle [DecayPlane/NeutralPion method]                   #
    # ------------------------------------------------------------------------------- #
    def geta1vecsDP(self, p4_os_pi, p4_ss1_pi, p4_ss2_pi):
        #print(f"{p4_ss1_pi.pdgId}, {p4_ss2_pi.pdgId}")
        Minv1 = (p4_os_pi + p4_ss1_pi).mass
        Minv2 = (p4_os_pi + p4_ss2_pi).mass
        
        Pi = ak.where(np.abs(0.77526-Minv1) < np.abs(0.77526-Minv2), p4_ss1_pi, p4_ss2_pi)
        PiZero = p4_os_pi
        
        return Pi, PiZero

    
    def getPhiCP_DP(self, PiPlus, PiMinus, PiZeroPlus, PiZeroMinus):

        ZMF = PiPlus + PiMinus
        boostv = ZMF.boostvec
        # minus side
        PiMinus_ZMF = PiMinus.boost(boostv.negative())
        PiZeroMinus_ZMF = PiZeroMinus.boost(boostv.negative())
        vecPiMinus = PiMinus_ZMF.pvec.unit
        vecPiZeroMinus = PiZeroMinus_ZMF.pvec.unit
        print(f"vecPiZeroMinus: {vecPiZeroMinus}")
        print(f"vecPiMinus: {vecPiMinus}")
        vecPiZeroMinustransv = (vecPiZeroMinus - vecPiMinus*(vecPiMinus.dot(vecPiZeroMinus))).unit
        # plus side
        PiPlus_ZMF = PiPlus.boost(boostv.negative())
        PiZeroPlus_ZMF = PiZeroPlus.boost(boostv.negative())
        vecPiPlus = PiPlus_ZMF.pvec.unit
        vecPiZeroPlus = PiZeroPlus_ZMF.pvec.unit
        print(f"vecPiZeroPlus: {vecPiZeroPlus}")
        print(f"vecPiPlus: {vecPiPlus}")
        vecPiZeroPlustransv = (vecPiZeroPlus - vecPiPlus*(vecPiPlus.dot(vecPiZeroPlus))).unit
        # Y variable
        Y1 = (PiMinus.energy - PiZeroMinus.energy)/(PiMinus.energy + PiZeroMinus.energy)
        Y2 = (PiPlus.energy - PiZeroPlus.energy)/(PiPlus.energy + PiZeroPlus.energy)
        Y  = Y1*Y2
        # angle
        acop_DP_1 = np.arccos(vecPiZeroPlustransv.dot(vecPiZeroMinustransv))
        sign_DP   = vecPiMinus.dot(vecPiZeroPlustransv.cross(vecPiZeroMinustransv))
        sign_mask = sign_DP < 0.0 
        acop_DP_2 = ak.where(sign_mask, 2*np.pi - acop_DP_1, acop_DP_1)
        Y_mask    = Y < 0.0
        acop_DP_3 = ak.where(Y_mask, acop_DP_2 + np.pi, acop_DP_2)
        mask      = Y_mask & (acop_DP_3 > 2*np.pi)
        acop_DP   = ak.where(mask, acop_DP_3 - 2*np.pi, acop_DP_3)

        return acop_DP

    
    def comp_PhiCP_DP(self):
        """
        Acoplanarity angle in decay plane method
        """
        PiMinus = None
        PiZeroMinus = None
        PiPlus = None
        PiZeroPlus = None

        if self.cat == "rhorho":
            evtdict = self.get_evtinfo_rhorho()

            PiMinus = copy.deepcopy(evtdict.taum.p4_pi)
            PiZeroMinus = copy.deepcopy(evtdict.taum.p4_pi0)
            PiPlus = copy.deepcopy(evtdict.taup.p4_pi)
            PiZeroPlus = copy.deepcopy(evtdict.taup.p4_pi0)

        elif self.cat == "a1rho":
            evtdict1 = self.get_evtinfo_a1rho_leg1()
            PiPlus_1 = copy.deepcopy(evtdict1.taup.p4_pi)
            PiZeroPlus_1 = copy.deepcopy(evtdict1.taup.p4_pi0)
            PiMinus_1, PiZeroMinus_1 = self.geta1vecsDP(evtdict1.taum.p4_os_pi,
                                                        evtdict1.taum.p4_ss1_pi,
                                                        evtdict1.taum.p4_ss2_pi) 

            evtdict2 = self.get_evtinfo_a1rho_leg2()
            PiMinus_2 = copy.deepcopy(evtdict2.taum.p4_pi)
            PiZeroMinus_2 = copy.deepcopy(evtdict2.taum.p4_pi0)
            PiPlus_2, PiZeroPlus_2 = self.geta1vecsDP(evtdict2.taup.p4_os_pi,
                                                      evtdict2.taup.p4_ss1_pi,
                                                      evtdict2.taup.p4_ss2_pi) 

            PiMinus = ak.concatenate([PiMinus_1, PiMinus_2], axis=0)
            PiZeroMinus = ak.concatenate([PiZeroMinus_1, PiZeroMinus_2], axis=0)
            PiPlus = ak.concatenate([PiPlus_1, PiPlus_2], axis=0)
            PiZeroPlus = ak.concatenate([PiZeroPlus_1, PiZeroPlus_2], axis=0)
            
        elif self.cat == "a1a1":
            evtdict = self.get_evtinfo_a1a1()
            PiMinus, PiZeroMinus = self.geta1vecsDP(evtdict.taum.p4_os_pi,
                                                    evtdict.taum.p4_ss1_pi,
                                                    evtdict.taum.p4_ss2_pi)
            PiPlus, PiZeroPlus   = self.geta1vecsDP(evtdict.taup.p4_os_pi,
                                                    evtdict.taup.p4_ss1_pi,
                                                    evtdict.taup.p4_ss2_pi)
        else:
            raise RuntimeWarning (f"{self.cat} is not suitable for DP method")

        phicp = self.getPhiCP_DP(PiPlus, PiMinus, PiZeroPlus, PiZeroMinus)
        plotit(arrlist=[ak.ravel(phicp).to_numpy()], bins=9)

        return phicp


    
    # ------------------------------------------------------------------------------- #
    #                       PhiCP angle [Impact parameter method]                     #
    # ------------------------------------------------------------------------------- #
    def getIP(self, p4, p3):
        dirvec = p4.pvec
        proj = p3.dot(dirvec.unit)
        return p3 - dirvec*proj


    def comp_PhiCP_IP(self):
        if self.cat == "pipi":
            boostvec, p4_taum_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest = self.get_evtinfo_pipi()
            
        else:
            raise RuntimeError(f"{self.cat} is not suitable for DP method")


        #plotit(arrlist=[ak.ravel(acop_IP).to_numpy()], bins=9)
        #return acop_IP
        pass
