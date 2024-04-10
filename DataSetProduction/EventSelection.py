# Preprocess data
import os
import glob
import yaml
import types
import uproot
import vector
import fnmatch
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
from coffea.nanoevents.methods import vector
from tqdm.notebook import tqdm
import itertools
from prettytable import PrettyTable
from typing import Optional
from PhiCPComp import PhiCPComp
from util import *


class SelectEvents:
    def __init__(self,
                 events: ak.Array = None,
                 tau1dm: Optional[int] = 1,
                 tau2dm: Optional[int] = 1,
                 isForTrain: Optional[bool] = True,
                 howtogetpizero: Optional[str]="simple_IC") -> None:
        """
          A class to select events
          arguments:
            events: events array
            tau1dm: decay mode of one tau
            tau2dm: decay mode of other tau
            isForTrain: selection includes the gentau selection
                        which will decide tau selection as well
            howtogetpizero: selection pizero from the tau decay products
        """
        self.events = events
        self.tau1dm = tau1dm
        self.tau2dm = tau2dm
        self.isForTrain = isForTrain
        self.howtogetpizero = howtogetpizero
        if self.howtogetpizero not in ["simple_IC", "simple_MB", "XGB_MB"]:
            raise RuntimeError(f"{self.howtogetpizero} is not mentioned as recommended")
        
        
    def muon_selection(self) -> tuple[dict, ak.Array]:
        muos = self.events.Muon
        sorted_indices = ak.argsort(muos.pt, axis=-1, ascending=False)
        muo_mask = (
            (muos.pt > 20) &
            (np.abs(muos.eta) < 2.4) &
            (abs(muos.dxy) < 0.045) &
            (abs(muos.dz) < 0.2) &
            (muos.mediumId == 1)
        )
        sel_muo_indices = sorted_indices[muo_mask[sorted_indices]]
        SelectionResult = {
            "steps": {
                "muon_selection: no_muon": ak.num(sel_muo_indices, axis=1) == 0
            },
        }
        print("muon selection done")
        return SelectionResult, sel_muo_indices

    

    def electron_selection(self, mu_idxs: Optional[ak.Array]=None) -> tuple[dict, ak.Array]:
        eles = self.events.Electron
        sorted_indices = ak.argsort(eles.pt, axis=-1, ascending=False)
        ele_mask = (
            (eles.pt > 25) &
            (np.abs(eles.eta) < 2.5) &
            (abs(eles.dxy) < 0.045) &
            (abs(eles.dz) < 0.2) &
            (eles.mvaIso_WP80 == 1)
        )
        if mu_idxs is not None:
            ele_mask = ele_mask & ak.all(eles.metric_table(self.events.Muon[mu_idxs]) > 0.2, axis=2)

        sel_ele_indices = sorted_indices[ele_mask[sorted_indices]]
        SelectionResult = {
            "steps": {
                "electron_selection: no_electron": ak.num(sel_ele_indices, axis=1) == 0
            },
        }
        print("electron selection done")
        return SelectionResult, sel_ele_indices

    

    def tau_selection(self,
                      ele_idxs: Optional[ak.Array]=None,
                      mu_idxs: Optional[ak.Array]=None,
                      gentaus: Optional[ak.Array] = None) -> tuple[dict, ak.Array]:
        taus = self.events.Tau
        sorted_indices = ak.argsort(taus.pt, axis=-1, ascending=False)
        mask_taudm = None
        tau_mask = (
            (taus.pt > 20)
            & (abs(taus.eta) < 2.3)
            & (abs(taus.dz) < 0.2)
            & ((taus.charge == 1) | (taus.charge == -1))
            & (taus.idDecayModeNewDMs)
            & ((taus.decayModePNet == 0)
               | (taus.decayModePNet == 1)
               | (taus.decayModePNet == 2)
               | (taus.decayModePNet == 10)
               | (taus.decayModePNet == 11))
            & (taus.idDeepTau2017v2p1VSjet >= 4)
            & (taus.idDeepTau2017v2p1VSe >= 2)
            & (taus.idDeepTau2017v2p1VSmu >= 1)
        )
        if ele_idxs is not None:
            tau_mask = tau_mask & ak.all(taus.metric_table(self.events.Electron[ele_idxs]) > 0.2, axis=2)
        if mu_idxs is not None:
            tau_mask = tau_mask & ak.all(taus.metric_table(self.events.Muon[mu_idxs]) > 0.2, axis=2)

        if self.tau1dm is not None and self.tau2dm is not None:
            tau_mask = tau_mask & ((taus.decayModePNet == self.tau1dm) | (taus.decayModePNet == self.tau2dm))

        sel_tau_indices = sorted_indices[tau_mask[sorted_indices]]
        taus = self.events.Tau[sel_tau_indices]
        two_taus_closest_gentaus_evt_mask = self.events.event >= 0
        if self.isForTrain and gentaus is not None:
            _taus = gentaus.nearest(taus, threshold=0.4)
            two_taus_closest_gentaus_evt_mask = (ak.num(_taus, axis=1) == 2) & (ak.sum(_taus.charge, axis=1) == 0)

        SelectionResult = {
            "steps": {
                "tau_selection: at_least_2_taus": ak.num(sel_tau_indices, axis=1) == 2,
                "tau_selection: opposite_signs": ak.sum(taus.charge, axis=1) == 0,
                "tau_selection: closest to gentaus": two_taus_closest_gentaus_evt_mask,
            },
        }
        print("tau selection done")
        return SelectionResult, sel_tau_indices



    def jet_selection(self,
                      ele_idxs: Optional[ak.Array]=None,
                      mu_idxs : Optional[ak.Array]=None,
                      tau_idxs: Optional[ak.Array]=None) -> tuple[dict, ak.Array]:
        jets = self.events.Jet
        sorted_indices = ak.argsort(jets.pt, axis=-1, ascending=False)
        jet_mask = (
            (jets.pt > 25)
            & (abs(jets.eta) < 4.7)
            & (jets.jetId == 6)
            & (
                (jets.pt >= 50.0) | (jets.puId == 4)
            )
        )
        if ele_idxs is not None:
            jet_mask = jet_mask & ak.all(jets.metric_table(self.events.Electron[ele_idxs]) > 0.2, axis=2)
        if mu_idxs is not None:
            jet_mask = jet_mask & ak.all(jets.metric_table(self.events.Muon[mu_idxs]) > 0.2, axis=2)
        if tau_idxs is not None:
            jet_mask = jet_mask & ak.all(jets.metric_table(self.events.Tau[tau_idxs]) > 0.2, axis=2)

        sel_jet_indices = sorted_indices[jet_mask[sorted_indices]]
        SelectionResult = {
            "steps": {},
        }
        print("jet selection done")
        return SelectionResult, sel_jet_indices



    def gentauandprods_selection(self):
        GenPart = self.events.GenPart
        # masks to select gen tau+ and tau-
        isgentau = ((np.abs(GenPart.pdgId) == 15)
                    & (GenPart.hasFlags(["isPrompt","isLastCopy"]))
                    & (GenPart.status == 2)
                    & (GenPart.pt >= 10)
                    & (np.abs(GenPart.eta) <= 2.3))
        
        # get gen tau+ and tau- objects from events GenParts
        gentau = GenPart[isgentau]

        # get tau mother indices
        gentau_momidx    = gentau.distinctParent.genPartIdxMother
        is_gentau_from_h = (GenPart[gentau_momidx].pdgId == 25) & (GenPart[gentau_momidx].status == 22)
        gentau_momidx    = gentau_momidx[is_gentau_from_h]
    
        # get tau products
        gentau_children = gentau.distinctChildren
        
        
        SelectionResult = {
            "steps": {
                "gentau_selection: has 2 gentau": (ak.num(gentau.pdgId, axis=1) == 2),
                "gentau_selection: tau+ and tau-": (ak.sum(gentau.pdgId, axis=1) == 0),
                "gentau_selection: two moms are h": (ak.num(gentau_momidx, axis=1) == 2),
            },
        }
        print("gentaus and products selection done")
        return SelectionResult, gentau, gentau_children


    def prod_selection(self, evt_dict, tau_sel_idx):
        events   = evt_dict["events"]
        taus     = evt_dict["det_taus"]
        tauprods = evt_dict["det_tauprods"]
        gentaus  = evt_dict["gen_taus"]
        gentauprods = evt_dict["gen_tauprods"]
        jets     = evt_dict["jets"]

        gentaus["px"] = gentaus.px
        gentaus["py"] = gentaus.py
        gentaus["pz"] = gentaus.pz
        gentaus["DM"] = genDM(gentauprods)

        gentauprods["px"] = gentauprods.px
        gentauprods["py"] = gentauprods.py
        gentauprods["pz"] = gentauprods.pz
        
        gentaunus = gentauprods[np.abs(gentauprods.pdgId) == 16]
        if self.isForTrain:
            assert ak.min(ak.num(gentaunus.pdgId, axis=1)) == 2

        # get tau prods for taus
        tau1_idx = tau_sel_idx[:,:1]
        tau1_idx_brdcst, tauprod_tauIdx = ak.broadcast_arrays(ak.firsts(tau1_idx,axis=1), tauprods.tauIdx)
        tau1prod_mask = tauprod_tauIdx == tau1_idx_brdcst
        tau2_idx = tau_sel_idx[:,1:2]
        tau2_idx_brdcst, _ = ak.broadcast_arrays(ak.firsts(tau2_idx,axis=1), tauprods.tauIdx)
        tau2prod_mask = tauprod_tauIdx == tau2_idx_brdcst

        tau1prods = tauprods[tau1prod_mask]
        tau2prods = tauprods[tau2prod_mask]
        tauprods_concat = ak.concatenate([tau1prods[:,None], tau2prods[:,None]], axis=1)
        
        # working on the tau prods
        is_pi = np.abs(tauprods_concat.pdgId) == 211
        is_gm = np.abs(tauprods_concat.pdgId) == 22

        has_atleast_one_pion  = ak.sum(is_pi, axis=-1) > 0
        has_atleast_one_gamma = ak.sum(is_gm, axis=-1) > 0
        has_one_pion          = ak.sum(is_pi, axis=-1) == 1

        #tau_prod_evt_mask = ak.sum((has_one_pion & has_atleast_one_gamma), axis=1) >= 2
        tau_prod_evt_mask = ak.sum(has_atleast_one_pion, axis=1) >= 2
        empty_indices = tauprods_concat[:,:,:0]

        pions    = ak.where(tau_prod_evt_mask, tauprods_concat[is_pi], empty_indices)
        pions_p4 = getp4(pions)

        photons = ak.where(tau_prod_evt_mask, tauprods_concat[is_gm], empty_indices)
        sorted_idx_photons = ak.argsort(photons.pt, ascending=False)
        photons = photons[sorted_idx_photons] # already sorted before
        # add photons p4 and later sum those p4s in the simple_IC method
        photons_p4 = getp4(photons)
        
        pizeros_p4 = getp4(photons[:,0:1]) # dummy
        if self.howtogetpizero == "simple_IC":
            eta_pizeros  = ak.firsts(photons_p4.eta, axis=-1)[:,:,None]
            phi_pizeros  = ak.firsts(photons_p4.phi, axis=-1)[:,:,None]
            mass_pizeros = 0.1396*ak.ones_like(phi_pizeros)
            #from IPython import embed; embed()
            #pt_pizeros   = ak.sum(photons.pt, axis=-1)[:,:,None]
            sum_photons_p4 = ak.sum(photons_p4, axis=-1)
            pt_pizeros   = sum_photons_p4.pt[:,:,None]

            pizeros_p4 = ak.zip(
                {
                    "pt"  : pt_pizeros,
                    "eta" : eta_pizeros,
                    "phi" : phi_pizeros,
                    "mass": mass_pizeros,
                    "pdgId": 111*ak.ones_like(mass_pizeros),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            )

        elif self.howtogetpizero == "simple_MB":
            pass

        elif self.howtogetpizero == "XGB_MB":
            pass

        else:
            raise RuntimeError("wrong method for howtogetpizero")

        evt_mask_1 = events.event >= 0
        evt_mask_2 = events.event >= 0
        if self.tau1dm == 1 and self.tau2dm == 1:
            evt_mask_1 = ak.sum((has_one_pion & has_atleast_one_gamma), axis=1) >= 2
            if self.isForTrain:
                #if self.tau1dm == 1 and self.tau2dm == 1:
                #evt_mask_1 = 
                one_prong = (gentaus.DM == 1)
                evt_mask_2 = ak.sum(one_prong, axis=1) == 2
        
        tauprod_results = {
            "steps": {
                #"tau prods with pi and pi0": tau_prod_evt_mask,
                "tau prods with pi and pi0": evt_mask_1,
                "gen DM 11"                : evt_mask_2,
            },
        }
        new_events_dict = evt_dict | {"gen_taunus": gentaunus,
                                      "det_tauprods_concat": tauprods_concat,
                                      "det_pions": pions_p4,
                                      "det_pizeros": pizeros_p4}
        # save nPions and nPhotons later :: MUST
        
        print("tau prods selection done")
        return tauprod_results, new_events_dict

    

    def apply_selections(self) -> dict:
        evt_cut_flow = dict()
        muon_results, muon_indices         = self.muon_selection()
        electron_results, electron_indices = self.electron_selection(muon_indices)
        gentau_results, gentaus, gentau_products = self.gentauandprods_selection()
        tau_results, tau_indices           = self.tau_selection(electron_indices, muon_indices, gentaus)
        jet_results, jet_indices           = self.jet_selection(electron_indices, muon_indices, tau_indices)

        event_sel_masks_fortrain = muon_results["steps"] | electron_results["steps"] | gentau_results["steps"] | tau_results["steps"] | jet_results["steps"]
        event_sel_masks_foreval  = muon_results["steps"] | electron_results["steps"] | tau_results["steps"] | jet_results["steps"]

        event_sel_masks = event_sel_masks_fortrain if self.isForTrain else event_sel_masks_foreval

        event_mask = np.abs(self.events.event) >= 0
        evt_cut_flow["Starts with"] = ak.sum(event_mask)
        for key, mask in event_sel_masks.items():
            event_mask = event_mask & mask
            evt_cut_flow[key] = ak.sum(event_mask)
            
        
        events = self.events[event_mask]
        gentaus = gentaus[event_mask]
        gentau_products = gentau_products[event_mask]
        sel_tau_indices = tau_indices[event_mask]
        taus = events.Tau[sel_tau_indices]
        sel_jet_indices = jet_indices[event_mask]
        jets = events.Jet[sel_jet_indices]

        evt_dict = {"events": events,
                    "det_taus": taus,
                    "det_tauprods": events.TauProd,
                    "gen_taus": gentaus,
                    "gen_tauprods": gentau_products,
                    "jets": jets}

        #self.print_cutflow(evt_cut_flow)

        # an extra selection based on tau decay products
        tauprod_results, evt_dict = self.prod_selection(evt_dict, sel_tau_indices)

        evt_mask = np.abs(events.event) >= 0
        for key, val in tauprod_results["steps"].items():
            evt_mask = evt_mask & val
            evt_cut_flow[key] = ak.sum(val)
            
        evt_dict = self.apply_evt_mask(evt_dict, evt_mask)
        
        self.print_cutflow(evt_cut_flow)
        self.inspect(evt_dict)
        
        #from IPython import embed; embed()
        return evt_dict

    

    def apply_evt_mask(self, evt_dict, evt_mask):
        return {key: val[evt_mask] for key, val in evt_dict.items()}


    def print_cutflow(self, event_cut_flow):
        x = PrettyTable()
        x.field_names = ["Selections", "nEvents Selected"]
        for key, val in event_cut_flow.items():
            x.add_row([key, val])
        print(x)
        

    def inspect(self, evt_dict):
        x = PrettyTable()
        x.field_names = ["Field", "Array"]
        print(" ===> Inspecting event dict ===> ")
        fields = evt_dict.keys()
        for key, val in evt_dict.items():
            x.add_row([key, type(val)])
        print(x)
