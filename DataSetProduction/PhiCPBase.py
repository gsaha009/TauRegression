import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from TComplex import TComplex
from typing import Optional
from coffea.nanoevents.methods import vector
from util import *


class PhiCPBase:
    def __init__(self, 
                 taum: ak.Array = None, 
                 taup: ak.Array = None, 
                 taum_decay: ak.Array = None, 
                 taup_decay: ak.Array = None) -> None:
        self.pid_pip =  211
        self.pid_pim = -211
        self.pid_kp  =  321
        self.pid_km  = -321
        self.pid_pi0 =  111
        self.pid_k0  =  311
        self.pid_ks0 =  310
        self.pid_kl0 =  130
        
        self.taum = taum
        self.taup = taup
        self.taum_decay = taum_decay
        self.taup_decay = taup_decay

        
    def selcols(self, mask):
        #mask = ak.flatten(mask)
        #dummy = self.taum
        #empty = dummy[...][...,:0]
        #from IPython import embed; embed()
        #taum = ak.where(mask, self.taum, empty)
        #taup = ak.where(mask, self.taup, empty)
        #taum_decay = ak.where(mask, self.taum_decay, empty)
        #taup_decay = ak.where(mask, self.taup_decay, empty)
        #return self.taum[mask], self.taup[mask], self.taum_decay[mask], self.taup_decay[mask]
        return self.taum, self.taup, self.taum_decay, self.taup_decay
        #return taum, taup, taum_decay, taup_decay

    
    def setboost(self, p4, bv):
        return p4.boost(bv.negative())


    # tau- -> pi- nu [DM: 0]
    def taum_pinu(self) -> ak.Array:
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 2) 
                 & (
                     (ak.sum(self.taum_decay.pdgId == self.pid_pim, axis=-1) == 1)   # pi-
                     | (ak.sum(self.taum_decay.pdgId == self.pid_km, axis=-1) == 1) # k-
                 ) & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1))    # nu
        return mask

    # tau+ -> pi+ nu [DM: 0]
    def taup_pinu(self) -> ak.Array:
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 2) 
                 & (
                     (ak.sum(self.taup_decay.pdgId == self.pid_pip, axis=-1) == 1)    # pi+
                     | (ak.sum(self.taup_decay.pdgId == self.pid_kp, axis=-1) == 1)  # k+
                 ) & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1))   # nu 
        return mask

    # tau- -> nu rho- -> nu pi- pi0 [DM: ]
    def taum_rho(self) -> ak.Array:
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 3) 
                & (
                    (ak.sum(self.taum_decay.pdgId == self.pid_pim, axis=-1) == 1)
                    | (ak.sum(self.taum_decay.pdgId == self.pid_km, axis=-1) == 1)
                )
                & (
                    (ak.sum(self.taum_decay.pdgId == self.pid_pi0, axis=-1) == 1)
                    | (ak.sum(self.taum_decay.pdgId == self.pid_k0, axis=-1) == 1)
                )
                & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1)
                )
        return mask
    
    # tau+ -> nu rho+ -> nu pi+ pi0 [DM: ]
    def taup_rho(self) -> ak.Array:
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 3)
                & (
                     (ak.sum(self.taup_decay.pdgId == self.pid_pip, axis=-1) == 1)
                     | (ak.sum(self.taup_decay.pdgId == self.pid_kp, axis=-1) == 1)
                 )
                & (
                    (ak.sum(self.taup_decay.pdgId == self.pid_pi0, axis=-1) == 1)
                    | (ak.sum(self.taup_decay.pdgId == self.pid_k0, axis=-1) == 1)
                )
                & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1)
                )
        return mask

    
    def taum_a1(self) -> ak.Array:
        # pid 311 will make things complicated
        mask = ((ak.num(self.taum_decay.pdgId, axis=-1) == 4) 
                 & (ak.sum(self.taum_decay.pdgId == -211, axis=-1) == 2)
                 & (ak.sum(self.taum_decay.pdgId == 211, axis=-1) == 1)
                 & (ak.sum(self.taum_decay.pdgId == 16, axis=-1) == 1))
        return mask

    
    def taup_a1(self) -> ak.Array:
        # pid 311 will make things complicated
        mask = ((ak.num(self.taup_decay.pdgId, axis=-1) == 4) 
                 & (ak.sum(self.taup_decay.pdgId == 211, axis=-1) == 2)
                 & (ak.sum(self.taup_decay.pdgId == -211, axis=-1) == 1)
                 & (ak.sum(self.taup_decay.pdgId == -16, axis=-1) == 1))
        return mask

    
    def getframe(self, p4arr1: ak.Array, p4arr2: ak.Array, verbose: Optional[bool] = True):
        frame = p4arr1 + p4arr2
        if verbose:
            print(f" --- pT  : {frame.pt}")
            print(f" --- eta : {frame.eta}")
            print(f" --- phi : {frame.phi}")
            print(f" --- mass: {frame.mass}")
        return frame

    
    # ------------------------------------------------------------------------------- #
    #                                     pi - pi                                     #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_pipi(self):
        print(" --- gethvecs_pipi ---")
        # get the mask to select tau-tau legs
        mask = self.taum_pinu() & self.taup_pinu()
        print(f"Selection of pi-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        taum, taup, taum_decay, taup_decay = self.selcols(mask)
        #from IPython import embed; embed()
        p4_taum = getp4(taum)
        #p4_taum = setp4_(p4_taum, 1.78)

        p4_taup = getp4(taup)
        #p4_taup = setp4_(p4_taup, 1.78)

        print("tau- in lab frame:")
        printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        printinfo(p4_taup, "p")  

        ispim = (np.abs(taum_decay.pdgId) == np.abs(self.pid_pim)) | (np.abs(taum_decay.pdgId) == np.abs(self.pid_km))
        ispip = (np.abs(taup_decay.pdgId) == np.abs(self.pid_pip)) | (np.abs(taup_decay.pdgId) == np.abs(self.pid_kp))
        isnu  = np.abs(taum_decay.pdgId) == 16
        
        p4_taum_nu = getp4(taum_decay[isnu])    # tau- nu
        p4_taum_pi = getp4(taum_decay[ispim])   # tau- pi-
        p4_taum_pi = setp4_(p4_taum_pi, 0.1396) # tau- pi-
        
        p4_taup_nu = getp4(taup_decay[isnu])    # tau+ nu
        p4_taup_pi = getp4(taup_decay[ispip])   # tau+ pi+
        p4_taup_pi = setp4_(p4_taup_pi, 0.1396) # tau+ pi+
        
        print("Frame is : pi+ + pi-")
        frame = self.getframe(p4_taum_pi, p4_taup_pi)
        printinfo(frame, "p")

        boostvec = frame.boostvec
        print("Boost vec: ")
        printinfoP3(boostvec)        

        p4_taum_hrest = self.setboost(p4_taum, boostvec)
        p4_taup_hrest = self.setboost(p4_taup, boostvec)

        outdict = {
            "boostv": boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4tau_rest": p4_taum_hrest,
                     "p4_pi": p4_taum_pi,
                     "p4_nu": p4_taum_nu},
            "taup": {"p4tau": p4_taup,
                     "p4tau_rest": p4_taup_hrest,
                     "p4_pi": p4_taup_pi,
                     "p4_nu": p4_taup_nu}
        }

        return obj(outdict)

    
    # ------------------------------------------------------------------------------- #
    #                                     pi - rho                                    #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_pirho_leg1(self):
        print(" --- gethvecs_rhopi --- ")
        mask = self.taum_rho() & self.taup_pinu()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        taum, taup, taum_decay, taup_decay = self.selcols(mask)
        
        p4_taum = getp4(taum)
        p4_taup = getp4(taup)
        print("tau- in lab frame:")
        printinfo(p4_taum, "p")
        print("tau+ in lab frame:")
        printinfo(p4_taup, "p")  

        ispim = (np.abs(taum_decay.pdgId) == 211) | (np.abs(taum_decay.pdgId) == 321)
        ispip = (np.abs(taup_decay.pdgId) == 211) | (np.abs(taup_decay.pdgId) ==  321)

        p4_taum_nu = getp4(taum_decay[np.abs(taum_decay.pdgId) == 16]) # tau- nu
        p4_taum_pi = getp4(taum_decay[ispim]) # tau- pi-
        ispi0 = (taum_decay.pdgId == 111) | (taum_decay.pdgId == 311) 
        p4_taum_pi0 = getp4(taum_decay[ispi0]) # tau- pi-

        p4_taup_nu = getp4(taup_decay[np.abs(taup_decay.pdgId) == 16]) # tau+ nu
        p4_taup_pi = getp4(taup_decay[ispip]) # tau+ pi+

        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        printinfoP3(h_boostvec)

        p4_taum_hrest = self.setboost(p4_taum, h_boostvec) # Get the tau- in the h rest frame
        p4_taup_hrest = self.setboost(p4_taup, h_boostvec) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        print("tau- in rest frame:")
        printinfo(p4_taum_hrest, "p")
        print("tau+ in rest frame:")
        printinfo(p4_taup_hrest, "p")  

        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_pi": p4_taum_pi,
                     "p4_pi0": p4_taum_pi0,
                     "p4_nu": p4_taum_nu},
            "taup": {"p4tau": p4_taup,
                     "p4tau_rest": p4_taup_hrest,
                     "p4_pi": p4_taup_pi,
                     "p4_nu": p4_taup_nu}
        }

        return obj(outdict)

        

    def get_evtinfo_pirho_leg2(self):
        print(" --- gethvecs_pirho_leg2 --- ")
        mask = self.taum_pinu() & self.taup_rho()
        print(f"Selection of pi-rho pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        taum, taup, taum_decay, taup_decay = self.selcols(mask)
        
        p4_taum = getp4(taum)
        p4_taup = getp4(taup)

        ispim = (np.abs(taum_decay.pdgId) == 211) | (np.abs(taum_decay.pdgId) == 321)
        ispip = (taup_decay.pdgId ==  211) | (taup_decay.pdgId ==  321)
        
        p4_taum_nu = getp4(taum_decay[taum_decay.pdgId == 16]) # tau- nu
        p4_taum_pi = getp4(taum_decay[ispim]) # tau- pi-

        p4_taup_nu = getp4(taup_decay[taup_decay.pdgId == -16]) # tau+ nu
        p4_taup_pi = getp4(taup_decay[ispip]) # tau+ pi+
        ispi0 = (taup_decay.pdgId == 111) | (taup_decay.pdgId == 311)
        p4_taup_pi0 = getp4(taup_decay[ispi0]) # tau+ pi+

        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        printinfoP3(h_boostvec)

        p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite

        outdict = {
            "boostv": h_boostvec,
            "taup": {"p4tau": p4_taup,
                     "p4_pi": p4_taup_pi,
                     "p4_pi0": p4_taup_pi0,
                     "p4_nu": p4_taup_nu},
            "taum": {"p4tau": p4_taum,
                     "p4tau_rest": p4_taum_hrest,
                     "p4_pi": p4_taum_pi,
                     "p4_nu": p4_taum_nu}
        }

        return obj(outdict)



    # ------------------------------------------------------------------------------- #
    #                                    rho - rho                                    #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_rhorho(self):
        print(" --- gethvecs_rhorho --- ")
        mask = self.taum_rho() & self.taup_rho()
        print(f"Selection of rho-rho pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        taum, taup, taum_decay, taup_decay = self.selcols(mask)
        
        p4_taum = getp4(taum)
        #p4_taum = setp4_(p4_taum, 1.78)

        p4_taup = getp4(taup)
        #p4_taup = setp4_(p4_taup, 1.78)
        
        ispim = (np.abs(taum_decay.pdgId) == 211) | (np.abs(taum_decay.pdgId) == 321)
        ispip = (np.abs(taup_decay.pdgId) == 211) | (np.abs(taup_decay.pdgId) == 321)        
        
        p4_taum_nu = getp4(taum_decay[np.abs(taum_decay.pdgId) == 16]) # tau- nu
        p4_taum_pi = getp4(taum_decay[ispim])                  # tau- pi-
        #p4_taum_pi = setp4_(p4_taum_pi, 0.1396)

        mask_pi0 = (taum_decay.pdgId == 111) | (taum_decay.pdgId == 311)
        p4_taum_pi0 = getp4(taum_decay[mask_pi0]) # tau- pi0
        #p4_taum_pi0 = setp4_(p4_taum_pi0, 0.1349)
        
        p4_taup_nu = getp4(taup_decay[np.abs(taup_decay.pdgId) == 16]) # tau+ nu
        p4_taup_pi = getp4(taup_decay[ispip]) # tau+ pi+
        #p4_taup_pi = setp4_(p4_taup_pi, 0.1396)
        
        mask_pi0 = (taup_decay.pdgId == 111) | (taup_decay.pdgId == 311)
        p4_taup_pi0 = getp4(taup_decay[mask_pi0]) # tau- pi0
        #p4_taup_pi0 = setp4_(p4_taup_pi0, 0.1349)
        
        p4_h = self.getframe(p4_taum, p4_taup)
        print("taup p4 + taum p4 ...")
        printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print("Higgs boost vec: ")
        printinfoP3(h_boostvec)

        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite

        #return h_boostvec, p4_taum, p4_taup, p4_taum_pi, p4_taup_pi, p4_taum_pi0, p4_taup_pi0, p4_taum_nu, p4_taup_nu, p4_taum_hrest, p4_taup_hrest
        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_pi": p4_taum_pi,
                     "p4_pi0": p4_taum_pi0,
                     "p4_nu": p4_taum_nu},
            "taup": {"p4tau": p4_taup,
                     "p4_pi": p4_taup_pi,
                     "p4_pi0": p4_taup_pi0,
                     "p4_nu": p4_taup_nu}
        }

        return obj(outdict)


        
    # ------------------------------------------------------------------------------- #
    #                                    a1 - a1                                      #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_a1a1(self):
        print(" \n\n--- get_evtinfo_a1a1 --- ")
        print(" --- get the masks to select a1-a1 events --- ")
        print(" --- then construct Lorentz vector --- ")
        # Prepare the collections first
        mask = self.taum_a1() & self.taup_a1()
        print(f"Selection of a1-a1 pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        #selcoldict = self.selcols(mask)
        taum, taup, taum_decay, taup_decay = self.selcols(mask)

        for i in range(5):
            print(f" {taum[i].pdgId}\t{taum_decay[i].pdgId}")
            print(f" {taum[i].pt}\t{taum_decay[i].pt}")
            print(f" {taum[i].status}\t{taum_decay[i].status}")
            print(f" {taup[i].pdgId}\t{taup_decay[i].pdgId}")
            print(f" {taup[i].pt}\t{taup_decay[i].pt}")
            print(f" {taup[i].status}\t{taup_decay[i].status}")
            print("\n")


        
        print(" --- taum ===>")
        p4_taum = getp4(taum, verbose=False)
        #p4_taum = setp4("PtEtaPhiMLorentzVector",
        #                p4_taum.pt,
        #                p4_taum.eta,
        #                p4_taum.phi,
        #                1.777+p4_taum.mass,
        #                p4_taum.pdgId,
        #                verbose=False)
        print(" --- taup ===>")
        p4_taup = getp4(taup, verbose=False)
        #p4_taup = setp4("PtEtaPhiMLorentzVector",
        #                p4_taup.pt,
        #                p4_taup.eta,
        #                p4_taup.phi,
        #                1.777+p4_taup.mass,
        #                p4_taup.pdgId,
        #                verbose=False)

        # tau- pi+
        print(" --- taum_os_pi (pi+) ===>")
        p4_taum_os_pi = getp4(taum_decay[taum_decay.pdgId == 211],
                              verbose=False)
        #p4_taum_os_pi = setp4("PtEtaPhiMLorentzVector",
        #                      p4_taum_os_pi.pt,
        #                      p4_taum_os_pi.eta,
        #                      p4_taum_os_pi.phi,
        #                      0.1396*ak.ones_like(p4_taum_os_pi.pt),
        #                      p4_taum_os_pi.pdgId,
        #                      verbose=False)
        # tau- pi-s
        p4_taum_ss_pi = getp4(taum_decay[taum_decay.pdgId == -211])
        # tau- pi1-
        print(" --- taum_ss_pi (pi1-) ===>")
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        #p4_taum_ss1_pi = setp4("PtEtaPhiMLorentzVector",
        #                       p4_taum_ss1_pi.pt,
        #                       p4_taum_ss1_pi.eta,
        #                       p4_taum_ss1_pi.phi,
        #                       0.1396*ak.ones_like(p4_taum_ss1_pi.pt),
        #                       p4_taum_ss1_pi.pdgId,
        #                       verbose=False)
        # tau- pi-2
        print(" --- taum_ss_pi (pi2-) ===>")
        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]
        #p4_taum_ss2_pi = setp4("PtEtaPhiMLorentzVector",
        #                       p4_taum_ss2_pi.pt,
        #                       p4_taum_ss2_pi.eta,
        #                       p4_taum_ss2_pi.phi,
        #                       0.1396*ak.ones_like(p4_taum_ss2_pi.pt),
        #                       p4_taum_ss2_pi.pdgId,
        #                       verbose=False)

        # tau+ pi-
        print(" --- taup_os_pi (pi-) ===>")
        p4_taup_os_pi = getp4(taup_decay[taup_decay.pdgId == -211],
                              verbose=False)
        #p4_taup_os_pi = setp4("PtEtaPhiMLorentzVector",
        #                      p4_taup_os_pi.pt,
        #                      p4_taup_os_pi.eta,
        #                      p4_taup_os_pi.phi,
        #                      0.1396*ak.ones_like(p4_taup_os_pi.pt),
        #                      p4_taup_os_pi.pdgId,
        #                      verbose=False)
        # tau+ pi+s
        p4_taup_ss_pi = getp4(taup_decay[taup_decay.pdgId == 211])
        # tau+ pi1+
        print(" --- taup_ss_pi (pi1+) ===>")
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        #p4_taup_ss1_pi = setp4("PtEtaPhiMLorentzVector",
        #                       p4_taup_ss1_pi.pt,
        #                       p4_taup_ss1_pi.eta,
        #                       p4_taup_ss1_pi.phi,
        #                       0.1396*ak.ones_like(p4_taup_ss1_pi.pt),
        #                       p4_taup_ss1_pi.pdgId,
        #                       verbose=False)
        # tau+ pi2+
        print(" --- taup_ss_pi (pi2+) ===>")
        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]
        #p4_taup_ss2_pi = setp4("PtEtaPhiMLorentzVector",
        #                       p4_taup_ss2_pi.pt,
        #                       p4_taup_ss2_pi.eta,
        #                       p4_taup_ss2_pi.phi,
        #                       0.1396*ak.ones_like(p4_taup_ss2_pi.pt),
        #                       p4_taup_ss2_pi.pdgId,
        #                       verbose=False)

        print(f" >>> tau- >>> children [os ss1 ss2] >>> ")
        for i in range(10):
            print(f" {p4_taum[i].pdgId}\t{p4_taum_os_pi[i].pdgId} {p4_taum_ss1_pi[i].pdgId} {p4_taum_ss2_pi[i].pdgId}")
            print(f" {p4_taum[i].pt}\t{p4_taum_os_pi[i].pt} {p4_taum_ss1_pi[i].pt} {p4_taum_ss2_pi[i].pt}")
            print(f" {p4_taup[i].pdgId}\t{p4_taup_os_pi[i].pdgId} {p4_taup_ss1_pi[i].pdgId} {p4_taup_ss2_pi[i].pdgId}")
            print(f" {p4_taup[i].pt}\t{p4_taup_os_pi[i].pt} {p4_taup_ss1_pi[i].pt} {p4_taup_ss2_pi[i].pt}")
            print("\n")
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)
        print(" --- higgs frame ===>")
        p4_h = self.getframe(p4_taum, p4_taup, verbose=True)

        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        print(" --- boostvec ===>")
        print(f" --- x: {h_boostvec.x}")
        print(f" --- y: {h_boostvec.y}")
        print(f" --- z: {h_boostvec.z}")
        
        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        #print(f"Opposite in phi?????? {p4_taum_hrest.delta_phi(p4_taup_hrest)}")
        #plotit(arrlist=[ak.ravel(p4_taum_hrest.delta_phi(p4_taup_hrest)).to_numpy()])

        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_os_pi": p4_taum_os_pi,
                     "p4_ss1_pi": p4_taum_ss1_pi,
                     "p4_ss2_pi": p4_taum_ss2_pi},
            "taup": {"p4tau": p4_taup,
                     "p4_os_pi": p4_taup_os_pi,
                     "p4_ss1_pi": p4_taup_ss1_pi,
                     "p4_ss2_pi": p4_taup_ss2_pi}
        }


        #return h_boostvec, p4_taum, p4_taup, p4_taum_os_pi, p4_taup_os_pi, p4_taum_ss1_pi, p4_taup_ss1_pi, p4_taum_ss2_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest
        print("\n\n")
        return obj(outdict)


    # ------------------------------------------------------------------------------- #
    #                                    a1 - pi                                      #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_a1pi_leg1(self):
        print(" --- gethvecs_a1pi_leg1 --- ")
        # Prepare the collections first
        mask = self.taum_a1() & self.taup_pinu()
        print(f"Selection of a1-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        #selcoldict = self.selcols(mask)
        taum, taup, taum_decay, taup_decay = self.selcols(mask)

        #ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)
        ispim = lambda col: ((col.pdgId == -211) | (col.pdgId == -311))
        ispip = lambda col: ((col.pdgId == 211) | (col.pdgId == 311))

        #p4_taum = getp4(selcoldict['taum'])
        #p4_taup = getp4(selcoldict['taup'])
        p4_taum = getp4(taum)
        p4_taup = getp4(taup)

        #p4_taum_os_pi = getp4(selcoldict['taum_decay'][ispip(selcoldict['taum_decay'])]) # tau- pi+
        #p4_taum_ss_pi = getp4(selcoldict['taum_decay'][ispim(selcoldict['taum_decay'])]) # tau- pi-
        p4_taum_os_pi = getp4(taum_decay[ispip(taum_decay)]) # tau- pi+
        p4_taum_ss_pi = getp4(taum_decay[ispim(taum_decay)]) # tau- pi-
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]

        #print(taum_nu_cat_tautau_pinu.pdgId)
        #print(taum_pi_cat_tautau_pinu.pdgId)
        
        #p4_taup_pi = getp4(selcoldict['taup_decay'][ispip(selcoldict['taup_decay'])]) # tau+ pi+
        #p4_taup_nu = getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau+ nu
        p4_taup_pi = getp4(taup_decay[ispip(taup_decay)])       # tau+ pi+
        p4_taup_nu = getp4(taup_decay[taup_decay.pdgId == -16]) # tau+ nu
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)

        p4_h = self.getframe(p4_taum, p4_taup)
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        #return h_boostvec, p4_taum, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taup_pi, p4_taum_hrest, p4_taup_hrest
        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_os_pi": p4_taum_os_pi,
                     "p4_ss1_pi": p4_taum_ss1_pi,
                     "p4_ss2_pi": p4_taum_ss2_pi},
            "taup": {"p4tau": p4_taup,
                     "p4_pi": p4_taup_pi}
        }
        return obj(outdict)



    def get_evtinfo_a1pi_leg2(self):
        print(" --- gethvecs_a1pi_leg2 --- ")
        # Prepare the collections first
        mask = self.taup_a1() & self.taum_pinu()
        print(f"Selection of a1-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        #selcoldict = self.selcols(mask)
        taum, taup, taum_decay, taup_decay = self.selcols(mask)
        
        #ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        ispim = lambda col: ((col.pdgId == -211) | (col.pdgId == -311))
        ispip = lambda col: ((col.pdgId == 211) | (col.pdgId == 311))

        #p4_taum = getp4(selcoldict['taum'])
        #p4_taup = getp4(selcoldict['taup'])
        p4_taum = getp4(taum)
        p4_taup = getp4(taup)

        #p4_taup_os_pi = getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -211]) # tau- pi+
        #p4_taup_ss_pi = getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 211]) # tau- pi-
        p4_taup_os_pi = getp4(taup_decay[ispim(taup_decay)]) # tau- pi+
        p4_taup_ss_pi = getp4(taup_decay[ispip(taup_decay)]) # tau- pi-
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]

        #print(taum_nu_cat_tautau_pinu.pdgId)
        #print(taum_pi_cat_tautau_pinu.pdgId)
        
        p4_taum_pi = getp4(taum_decay[ispim(taum_decay)])      # tau+ pi+
        p4_taum_nu = getp4(taum_decay[taum_decay.pdgId == 16]) # tau+ nu
        
        #print(taup_nu_cat_tautau_pinu.pdgId)
        #print(taup_pi_cat_tautau_pinu.pdgId)

        p4_h = self.getframe(p4_taum, p4_taup)
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame

        #return h_boostvec, p4_taup, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_pi, p4_taum_hrest, p4_taup_hrest
        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_os_pi": p4_taum_os_pi,
                     "p4_ss1_pi": p4_taum_ss1_pi,
                     "p4_ss2_pi": p4_taum_ss2_pi},
            "taup": {"p4tau": p4_taup,
                     "p4_pi": p4_taup_pi}
        }
        return obj(outdict)


    # ------------------------------------------------------------------------------- #
    #                                     a1 - rho                                    #
    # ------------------------------------------------------------------------------- #
    def get_evtinfo_a1rho_leg1(self):
        print(" --- gethvecs_a1rho_leg1 --- ")
        print(" --- taum to a1, taup to rho --- ")
        mask = self.taum_a1() & self.taup_rho()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        #selcoldict = self.selcols(mask)
        taum, taup, taum_decay, taup_decay = self.selcols(mask)

        p4_taum = getp4(selcoldict['taum'])
        #p4_taum = setp4_(p4_taum, 1.777)

        p4_taup = getp4(selcoldict['taup'])
        #p4_taup = setp4_(p4_taup, 1777)

        #print("tau- in lab frame:")
        #printinfo(p4_taum, "p")
        #print("tau+ in lab frame:")
        #printinfo(p4_taup, "p")  

        #ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        #ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        ispim = lambda col: ((col.pdgId == -211) | (col.pdgId == -311))
        ispip = lambda col: ((col.pdgId == 211) | (col.pdgId == 311))
        
        p4_taum_os_pi = getp4(selcoldict['taum_decay'][ispip(selcoldict['taum_decay'])]) # tau- pi+
        p4_taum_os_pi = setp4_(p4_taum_os_pi, 0.1396)

        p4_taum_ss_pi = getp4(selcoldict['taum_decay'][ispim(selcoldict['taum_decay'])]) # tau- pi-
        p4_taum_ss1_pi = p4_taum_ss_pi[:,0:1]
        #p4_taum_ss1_pi = setp4_(p4_taum_ss1_pi, 0.1396)

        p4_taum_ss2_pi = p4_taum_ss_pi[:,1:2]
        #p4_taum_ss2_pi = setp4_(p4_taum_ss2_pi, 0.1396)

        
        p4_taup_nu = getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == -16]) # tau- nu
        p4_taup_pi = getp4(selcoldict['taup_decay'][ispip(selcoldict['taup_decay'])]) # tau- pi-
        #p4_taup_pi = setp4_(p4_taup_pi, 0.1396)

        p4_taup_pi0 = getp4(selcoldict['taup_decay'][selcoldict['taup_decay'].pdgId == 111]) # tau- pi-
        #p4_taup_pi0 = setp4_(p4_taup_pi0, 0.1349)

        p4_h = self.getframe(p4_taum, p4_taup)
        #print("taup p4 + taum p4 ...")
        #printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        #print("Higgs boost vec: ")
        #printinfoP3(h_boostvec)

        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        #print("tau- in rest frame:")
        #printinfo(p4_taum_hrest, "p")
        #print("tau+ in rest frame:")
        #printinfo(p4_taup_hrest, "p")  

        #return h_boostvec, p4_taup, p4_taup_pi, p4_taup_pi0, p4_taup_nu, p4_taum_os_pi, p4_taum_ss1_pi, p4_taum_ss2_pi, p4_taum_hrest, p4_taup_hrest

        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_os_pi": p4_taum_os_pi,
                     "p4_ss1_pi": p4_taum_ss1_pi,
                     "p4_ss2_pi": p4_taum_ss2_pi},
            "taup": {"p4tau": p4_taup,
                     "p4_pi": p4_taup_pi,
                     "p4_pi0": p4_taup_pi0,
                     "p4_nu": p4_taup_nu}
        }
        
        return obj(outdict)


    def get_evtinfo_a1rho_leg2(self):
        print(" --- gethvecs_a1rho_leg2 --- ")
        print(" --- taum to rho, taup to a1 --- ")
        mask = self.taum_rho() & self.taup_a1()
        print(f"Selection of rho-pi pair [mask]: {mask}")
        print(f"n total events: {ak.count(mask)}")
        print(f"n selected events: {ak.sum(mask)}")
        selcoldict = self.selcols(mask)
        
        p4_taum = getp4(selcoldict['taum'])
        #p4_taum = setp4_(p4_taum, 1.777)
        
        p4_taup = getp4(selcoldict['taup'])
        #p4_taup = setp4_(p4_taup, 1.777)

        #ispim = (selcoldict['taum_decay'].pdgId == -211) | (selcoldict['taum_decay'].pdgId == -311)
        #ispip = (selcoldict['taup_decay'].pdgId == 211) | (selcoldict['taup_decay'].pdgId == 311)

        ispim = lambda col: ((col.pdgId == -211) | (col.pdgId == -311))
        ispip = lambda col: ((col.pdgId == 211) | (col.pdgId == 311))
        
        p4_taum_nu = getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 16]) # tau- nu
        p4_taum_pi = getp4(selcoldict['taum_decay'][ispim(selcoldict['taum_decay'])]) # tau- pi-
        #p4_taum_pi = setp4_(p4_taum_pi, 0.1396)

        p4_taum_pi0 = getp4(selcoldict['taum_decay'][selcoldict['taum_decay'].pdgId == 111]) # tau- pi-
        #p4_taum_pi0 = setp4_(p4_taum_pi0, 0.1349)

        p4_taup_os_pi = getp4(selcoldict['taup_decay'][ispim(selcoldict['taup_decay'])]) # tau+ pi-
        #p4_taup_os_pi = setp4_(p4_taup_os_pi, 0.1396)

        p4_taup_ss_pi = getp4(selcoldict['taup_decay'][ispip(selcoldict['taup_decay'])]) # tau+ pi+
        p4_taup_ss1_pi = p4_taup_ss_pi[:,0:1]
        #p4_taup_ss1_pi = setp4_(p4_taup_ss1_pi, 0.1396)

        p4_taup_ss2_pi = p4_taup_ss_pi[:,1:2]
        #p4_taup_ss2_pi = setp4_(p4_taup_ss2_pi, 0.1396)

        p4_h = self.getframe(p4_taum, p4_taup)
        #print("taup p4 + taum p4 ...")
        #printinfo(p4_h, "p")
        h_boostvec = p4_h.boostvec # Get the boost of Higgs
        #print("Higgs boost vec: ")
        #printinfoP3(h_boostvec)

        #p4_taum_hrest = p4_taum.boost(h_boostvec.negative()) # Get the tau- in the h rest frame
        #p4_taup_hrest = p4_taup.boost(h_boostvec.negative()) # Get the tau+ in the h rest frame
        #print(p4_taum_hrest.delta_phi(p4_taup_hrest)) # check if they are opposite
        #print("tau- in rest frame:")
        #printinfo(p4_taum_hrest, "p")
        #print("tau+ in rest frame:")
        #printinfo(p4_taup_hrest, "p")  

        #return h_boostvec, p4_taum, p4_taum_pi, p4_taum_pi0, p4_taum_nu, p4_taup_os_pi, p4_taup_ss1_pi, p4_taup_ss2_pi, p4_taum_hrest, p4_taup_hrest
        outdict = {
            "boostv": h_boostvec,
            "taum": {"p4tau": p4_taum,
                     "p4_pi": p4_taum_pi,
                     "p4_pi0": p4_taum_pi0,
                     "p4_nu": p4_taum_nu},
            "taup": {"p4tau": p4_taup,
                     "p4_os_pi": p4_taup_os_pi,
                     "p4_ss1_pi": p4_taup_ss1_pi,
                     "p4_ss2_pi": p4_taup_ss2_pi},
        }
        return obj(outdict)




    
        

    def getP4_ttrf_hf_trf(self, p4, boost_ttrf, r, n, k, boost_trf):
        p4_ttrf = self.getP4_rf(p4, boost_ttrf)
        p4_hf   = self.getP4_hf(p4_ttrf, r, n, k)
        p4_trf  = self.getP4_rf(p4_hf, boost_trf)

        return p4_trf
        

    def getHelicityAxes(self, boost_ttrf, tauP4_ttrf, p4_h, p4tauLF):
        # get helicity basis: r, n, k
        # get_localCoordinateSystem(evt.tauMinusP4(), &higgsP4, &boost_ttrf, hAxis_, collider_, r, n, k, verbosity_, cartesian_);
        # https://github.com/veelken/tautau-Entanglement/blob/main/src/PolarimetricVectorAlgoThreeProng0Pi0.cc#L141
        print("  --- getHelicityAxes ---")

        k = self.get_k(p4tauLF, boost_ttrf)
        h = self.get_h_higgsAxis(p4_h, boost_ttrf)
        #h = self.get_h_beamAxis(boost_ttrf)
        r = self.get_r(k, h)
        n = self.get_n(k, r)

        print(f"n.r: {n.dot(r)}")
        print(f"n.k: {n.dot(k)}")
        print(f"r.k: {r.dot(k)}")

        print(f"  plot: n.r, n.k, r.k")
        plotit(arrlist=[ak.ravel(n.dot(r)).to_numpy(),
                        ak.ravel(n.dot(k)).to_numpy(),
                        ak.ravel(r.dot(k)).to_numpy()],
               dim=(1,3))
        
        return r, n, k
        

    def get_k(self, taup4_LF, boost_ttrf):
        print("     --- get_k ---")
        # -------------------------------- T E S T --------------------------------- #
        k = taup4_LF.boost(boost_ttrf.negative())
        #k = taup4_LF.boost(taup4_LF.boostvec.negative())
        out = k.pvec.unit
        printinfoP3(out)

        return out

    
    def get_h_higgsAxis(self, recoilP4, boost_ttrf):
        # CV: this code does not depend on the assumption that the tau pair originates from a Higgs boson decay;
        # it also works for tau pairs originating from Z/gamma* -> tau+ tau- decays
        print("     --- get_h_higgsAxis ---")
        sf = 1.01
        higgsPx = sf*recoilP4.px
        higgsPy = sf*recoilP4.py
        higgsPz = sf*recoilP4.pz
        higgsE = np.sqrt(higgsPx**2 + higgsPy**2 + higgsPz**2 + recoilP4.mass2)

        higgsP4 = setp4("LorentzVector", higgsPx, higgsPy, higgsPz, higgsE)
        printinfo(higgsP4)
        printinfo(higgsP4, "pt")
        
        #higgsP4_ttrf = higgsP4.boost(boost_ttrf.negative())
        #h = higgsP4_ttrf.pvec.unit
        # -------------------------------- T E S T --------------------------------- #
        h = higgsP4.pvec.unit
        print("     boost: ")

        printinfoP3(h)
        
        return h


    def get_h_beamAxis(self, boost_ttrf):
        # https://github.com/veelken/tautau-Entanglement/blob/main/src/get_localCoordinateSystem.cc#L32
        print("     --- get_h_beamAxis ---")
        dummybeam = ak.ones_like(boost_ttrf.x)
        beamE = 6500*dummybeam             # 6.5 TeV
        mBeamParticle = 0.938272*dummybeam # proton mass[GeV]

        beamPx = 0.0*dummybeam
        beamPy = 0.0*dummybeam
        beamPz = np.sqrt(beamE**2 - mBeamParticle**2)

        beamP4 = setp4("LorentzVector", beamPx, beamPy, beamPz, beamE)
        print("\t\t beamP4 before boost")
        printinfo(beamP4)
        
        beamP4 = beamP4.boost(boost_ttrf.negative())
        print("\t\t beamP4 after boost")
        printinfo(beamP4)
        
        h = beamP4.pvec.unit

        print("     boost: ")
        printinfoP3(h)
        
        return h

    
    
    def get_r(self, k, h):
        print("     --- get_r ---")
        costheta = k.dot(h)
        sintheta = np.sqrt(1. - costheta*costheta)
        plotit(arrlist=[ak.ravel(costheta).to_numpy(),
                        ak.ravel(sintheta).to_numpy()], dim=(1,2))
        r = (h - k*costheta)*(1./sintheta)

        printinfoP3(r)
        return r


    def get_n(self, k, r):
        print(f"     --- get_n ---")
        n = r.cross(k)
        printinfoP3(n)
        
        return n

    def getP4_rf(self, p4, boostv):
        print("   --- getP4_rf ---")
        out = p4.boost(boostv.negative())
        printinfo(out)
        printinfo(out, "pt", False)
        return out

    
    def getP4_hf(self, p4, r, n, k):
        # CV: rotate given four-vector to helicity frame (hf)
        print("     --- getP4_hf ---")
        p3 = p4.pvec
        pr = p3.dot(r)
        pn = p3.dot(n)
        pk = p3.dot(k)

        out = setp4("LorentzVector", pr, pn, pk, p4.energy)

        printinfo(out)
        return out
