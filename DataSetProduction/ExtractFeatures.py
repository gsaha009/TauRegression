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
import itertools

from typing import Optional
from util import *
from PhiCPComp import PhiCPComp



class FeatureExtraction:
    def __init__(self,
                 events: ak.Array,
                 taus: ak.Array, 
                 tauprods: ak.Array,
                 gentaus: ak.Array,
                 gentauprods: ak.Array,
                 gentaunus: ak.Array,
                 jets: ak.Array,
                 isnorm: Optional[bool]=False,
                 isForTrain: Optional[bool]=False):
        self.events = events
        self.taus = taus
        self.tau_feats = ["pt", "ptCorrPNet", "eta", "phi", "mass", "dxy", "dz", 
                          "rawDeepTau2018v2p5VSe", "rawDeepTau2018v2p5VSjet", "rawDeepTau2018v2p5VSmu", 
                          "rawIsodR03", "rawMVAnewDM2017v2", "leadTkPtOverTauPt",
                          "rawPNetVSe", "rawPNetVSjet", "rawPNetVSmu", 
                          "charge@cat", "decayMode@cat", "decayModePNet@cat",
                          "probDM0PNet", "probDM10PNet", "probDM11PNet", "probDM1PNet", "probDM2PNet"]
        self.tau1prods = ak.firsts(tauprods[:,:1], axis=1)
        self.tau2prods = ak.firsts(tauprods[:,1:2], axis=1)
        self.gentaus = gentaus
        self.gentauprods = gentauprods
        self.gentaunus = gentaunus
        self.jets = jets
        self.jet_feats = ["pt", "eta", "phi", "mass", "btagPNetB", "btagDeepFlavB"]
        self.met = events.MET
        self.isnorm = isnorm
        self.isForTrain = isForTrain


    def getmanuals(self):
        taus      = self.taus
        tau1prods = self.tau1prods
        tau2prods = self.tau2prods
        met       = self.met

        tau1prods_p4 = getp4(tau1prods)
        tau2prods_p4 = getp4(tau2prods)
        taus_p4      = getp4(taus)
        
        #print(ak.to_list(tauprods.tauIdx), tauprods_p4.type, taus_p4.type)
        npf_tau_1 = ak.num(tau1prods.pdgId, axis=1)[:,None]
        npf_tau_2 = ak.num(tau2prods.pdgId, axis=1)[:,None]
        #print(f"npf_tau_1: {npf_tau_1}")
        #print(f"npf_tau_2: {npf_tau_2}")
        
        pf_tau1_pt = ak.sum(tau1prods_p4, axis=1).pt[:,None]
        pf_tau2_pt = ak.sum(tau2prods_p4, axis=1).pt[:,None]
        #print(f"pf_tau1_pt: {pf_tau1_pt}")
        #print(f"pf_tau2_pt: {pf_tau2_pt}")
        
        ptfrac_tau_1 = pf_tau1_pt/taus[:,:1].pt
        ptfrac_tau_2 = pf_tau2_pt/taus[:,1:2].pt
        #print(f"ptfrac_tau_1: {ptfrac_tau_1}")
        #print(f"ptfrac_tau_2: {ptfrac_tau_2}")
        
        
        pf_tau1_dphi = ak.firsts(taus_p4[:,0:1].metric_table(tau1prods_p4, metric=lambda a, b: a.delta_phi(b)), axis=1)
        pf_tau2_dphi = ak.firsts(taus_p4[:,1:2].metric_table(tau2prods_p4, metric=lambda a, b: a.delta_phi(b)), axis=1)
        #print(f"pf_tau1_dphi: {pf_tau1_dphi}")
        #print(f"pf_tau2_dphi: {pf_tau2_dphi}")
        pf_tau1_dphi_pt_frac = ak.sum(tau1prods_p4.pt*pf_tau1_dphi, axis=1)[:,None]/pf_tau1_pt
        pf_tau2_dphi_pt_frac = ak.sum(tau2prods_p4.pt*pf_tau2_dphi, axis=1)[:,None]/pf_tau2_pt
        #print(f"pf_tau1_dphi_pt_frac: {pf_tau1_dphi_pt_frac}")
        #print(f"pf_tau2_dphi_pt_frac: {pf_tau2_dphi_pt_frac}")
        
        pf_tau1_deta = ak.firsts(taus_p4[:,0:1].metric_table(tau1prods_p4, metric=lambda a, b: a.eta - b.eta), axis=1)
        pf_tau2_deta = ak.firsts(taus_p4[:,1:2].metric_table(tau2prods_p4, metric=lambda a, b: a.eta - b.eta), axis=1)
        #print(f"pf_tau1_deta: {pf_tau1_deta}")
        #print(f"pf_tau2_deta: {pf_tau2_deta}")
        pf_tau1_deta_pt_frac = ak.sum(pf_tau1_pt*pf_tau1_deta, axis=1)[:,None]/pf_tau1_pt
        pf_tau2_deta_pt_frac = ak.sum(pf_tau2_pt*pf_tau2_deta, axis=1)[:,None]/pf_tau2_pt
        #print(f"pf_tau1_deta_pt_frac: {pf_tau1_deta_pt_frac}")
        #print(f"pf_tau2_deta_pt_frac: {pf_tau2_deta_pt_frac}")
        
        concat_array = ak.to_numpy(ak.concatenate([npf_tau_1, npf_tau_2, ptfrac_tau_1, ptfrac_tau_2,
                                                   pf_tau1_dphi_pt_frac, pf_tau2_dphi_pt_frac,
                                                   pf_tau1_deta_pt_frac, pf_tau2_deta_pt_frac], axis=1))
        
        key_list = ["npf_tau_1", "npf_tau_2", "ptfrac_tau_1", "ptfrac_tau_2",
                    "pf_dphi_pt_frac_tau_1", "pf_dphi_pt_frac_tau_2",
                    "pf_deta_pt_frac_tau_1", "pf_deta_pt_frac_tau_2"]
        
        #print(f"manual shape: {concat_array.shape}")
        #print(np.mean(concat_array, axis=0))
        #print(np.std(concat_array, axis=0))
        if self.isnorm:
            concat_array = (concat_array - np.mean(concat_array, axis=0))/np.std(concat_array, axis=0)

        print(f"Total manual feats: {len(key_list)}")
        print(f"Node manual shape: {concat_array.shape}")


        print("Manual Features done")

        return concat_array, key_list


    # ----------------------------------------------------------------------------------------------------- #
    #                                            Node features                                              #
    # ----------------------------------------------------------------------------------------------------- #
    def getnodes(self, tag):
        objs = None
        featlist = None
        maxidx = 2
        if tag == "tau":
            objs = self.taus
            featlist = self.tau_feats
            maxidx = 2
        elif tag == "jet":
            objs = self.jets
            featlist = self.jet_feats
            maxidx = 15

        keys, feats = self.get_feats_per_obj(objs, featlist, maxidx, tag)
        print(f"Total node feats: {len(keys)}")
        print(f"Node feat shape: {feats.shape}")
        
        return feats, keys


    def get_feats_per_obj(self, col, featlist, maxidx, _tag):
        node_np = None
        keys_np = None
        for i, feat in enumerate(featlist):
            iscat = False
            if "@cat" in feat:
                iscat = True
                feat = feat.replace("@cat","")
        
            #tag = f"{feat}_{_tag}"
            var = None
            if feat not in col.fields:
                var = ak.zeros_like(col.pt)
            else:
                var = col[feat]
            var_keys, np_var = self.tonumpy(var, maxidx, iscat, feat, _tag) # feat: px, tag: tau
        
            if i == 0:
                node_np = np_var
                keys_np = var_keys
            else:
                node_np = np.concatenate((node_np, np_var), axis=1)
                keys_np = keys_np + var_keys

        return keys_np, node_np


    def tonumpy(self, arr, maxidx, iscat, feat, tag):
        keylist = []
        _uniques = {"charge": [-1., 1.], "decayMode": [0.,1.,10.,11.], "decayModePNet": [-1.,0.,1.,2.,10.,11.]}
        nptemp = None
        for idx in range(maxidx):
            #key = f"{tag}_{idx+1}"
            key = f"{feat}_{tag}_{idx+1}"
            print(key)
            temp = ak.to_numpy(ak.fill_none(ak.firsts(arr[:, idx : idx+1], axis=1), 0.0))
            #temp = ak.to_numpy(ak.fill_none(ak.firsts(arr[:, idx : idx+1], axis=1), 0.0))[:,None]
            #temp = temp[:,None]
            if iscat:
                print("categorical ---> ")
                print("first adding the original one ===> ")
                #temp = temp[:,None]
                #keylist.append(key)
                npones  = np.ones_like(temp)
                npzeros = np.zeros_like(temp)
                #uniques = np.unique(temp)
                uniques = np.array(_uniques[feat])
                print(f"\tuniques: {uniques}")
                catarray = None
                keylist.append(key)
                for i, val in enumerate(uniques):
                    #_key = f"{tag}_{val}_{idx+1}"
                    _key = f"{feat}_{int(val)}_{tag}_{idx+1}"
                    print(f"\t{_key}")
                    keylist.append(_key)
                    tempcatarray = np.where(temp == val, npones, npzeros)[:,None]
                    #print(f"\t{tempcatarray}")
                    if i == 0:
                        #catarray = tempcatarray
                        catarray = np.concatenate((temp[:,None],tempcatarray), axis=1)
                    else:
                        catarray = np.concatenate((catarray, tempcatarray), axis=1)
                temp = catarray
            elif self.isnorm and np.sum(temp) > 0.0:
                temp = ((temp - np.mean(temp))/np.std(temp))[:,None]
                keylist.append(key)
            else:
                temp = temp[:,None]
                keylist.append(key)

            #print(temp)
            if idx == 0:
                nptemp = temp
            else:
                nptemp = np.concatenate((nptemp, temp), axis=1)
            #print(temp)
            #keylist.append(key)
        return keylist, nptemp


    # ----------------------------------------------------------------------------------------------------- #
    #                                          Global features                                              #
    # ----------------------------------------------------------------------------------------------------- #        
    def global_var_to_keep(self):
        globalDict = {
            "pt_MET"           : {"coll": self.met, "fname": "pt"},
            "phi_MET"          : {"coll": self.met, "fname": "phi"},
            "covXX_MET"        : {"coll": self.met, "fname": "covXX"},
            "covXY_MET"        : {"coll": self.met, "fname": "covXY"},
            "covYY_MET"        : {"coll": self.met, "fname": "covYY"},
            "sumEt_MET"        : {"coll": self.met, "fname": "sumEt"},
            "significance_MET" : {"coll": self.met, "fname": "significance"},
            "nJet"             : {"coll": self.jets, "fname": "num"},
            "nPV"              : {"coll": self.events.PV, "fname": "npvsGood"},
            "HT"               : {"coll": self.jets.pt, "fname": "sum"},
            "MT_tau1_met"      : {"coll": [self.taus[:,:1], self.met], "fname": lambda x, y: get_MT(x,y)},
            "MT_tau2_met"      : {"coll": [self.taus[:,1:2], self.met], "fname": lambda x, y: get_MT(x,y)},
            "MT_tau1_tau2"     : {"coll": [self.taus[:,:1], self.taus[:,1:2]], "fname": lambda x, y: get_MT(x,y)},
            "MT_total"         : {"coll": [self.taus[:,:1], self.taus[:,1:2], self.met], "fname": lambda x, y, z: get_total_MT(x,y,z)},
        }
        return globalDict


    def getglobals(self, vardict):
        global_np = None
        global_feats = []
        for i, (key, val) in enumerate(vardict.items()):
            temp = None
            #print(f"variable: {key}")
            obj = val["coll"]
            col = val["fname"]
            #print(f"column: {col}")
            if col == "num":
                temp = ak.to_numpy(ak.fill_none(ak.num(obj, axis=1), 0.0))[:,None]
            elif col == "sum":
                temp = ak.to_numpy(ak.fill_none(ak.sum(obj, axis=1), 0.0))[:,None]
            elif isinstance(col, types.LambdaType):
                temp = ak.to_numpy(ak.fill_none(col(*obj), 0.0))
            else:
                assert col in obj.fields, f"feat must be in the fields"
                temp = ak.to_numpy(ak.fill_none(obj[col], 0.0))[:,None]
        
            #print(temp)

            if i == 0:
                global_np = temp
            else:
                global_np = np.concatenate((global_np, temp), axis=1)

            global_feats.append(key)

        if self.isnorm:
            global_np = (global_np - np.mean(global_np, axis=0))/np.std(global_np, axis=0)

        #data  = pd.DataFrame(global_np, columns=global_feats)
        #return data
        print(f"Total global feats: {len(global_feats)}")
        print(f"Global feat shape: {global_np.shape}")
        return global_np, global_feats


    # ----------------------------------------------------------------------------------------------------- #
    #                                            get targets                                                #
    # ----------------------------------------------------------------------------------------------------- #
    def gettargets(self):
        gentaus = self.gentaus
        gentaunus = self.gentaunus

        np_target_feats = ak.to_numpy(ak.zeros_like(self.events.event)[:,None][...,:0])
        target_keys = []

        if self.isForTrain:
            gentau_1_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].px, axis=1), 0.0))[:,None]
            gentau_1_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].py, axis=1), 0.0))[:,None]
            gentau_1_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].pz, axis=1), 0.0))[:,None]
            gentau_2_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].px, axis=1), 0.0))[:,None]
            gentau_2_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].py, axis=1), 0.0))[:,None]
            gentau_2_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].pz, axis=1), 0.0))[:,None]
            
            gentaunu_1_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].px, axis=1), 0.0))
            gentaunu_1_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].py, axis=1), 0.0))
            gentaunu_1_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].pz, axis=1), 0.0))
            gentaunu_2_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].px, axis=1), 0.0))
            gentaunu_2_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].py, axis=1), 0.0))
            gentaunu_2_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].pz, axis=1), 0.0))

            #from IPython import embed; embed()
            
            np_target_feats = np.concatenate((gentau_1_px, gentau_1_py, gentau_1_pz, 
                                              gentau_2_px, gentau_2_py, gentau_2_pz,
                                              gentaunu_1_px, gentaunu_1_py, gentaunu_1_pz, 
                                              gentaunu_2_px, gentaunu_2_py, gentaunu_2_pz), axis=1)
            target_keys = ["px_gentau_1", "py_gentau_1", "pz_gentau_1", 
                           "px_gentau_2", "py_gentau_2", "pz_gentau_2",
                           "px_gentaunu_1", "py_gentaunu_1", "pz_gentaunu_1", 
                           "px_gentaunu_2", "py_gentaunu_2", "pz_gentaunu_2"]
            
        print(f"Total target feats: {len(target_keys)}")
        print(f"Target feat shape: {np_target_feats.shape}")
        
        return np_target_feats, target_keys


    # keep it flexible so that it can be called from outside
    def extrafeats(self, tau1dm, tau2dm, pions, pizeros):
        print(" --- Saving extra feats >>>")
        _tauprods = ak.concatenate([pions,pizeros], axis=-1)


        events = self.events
        pions_tau1    = ak.firsts(pions[:,0:1])
        pions_tau2    = ak.firsts(pions[:,1:2])
        pizeros_tau1  = ak.firsts(pizeros[:,0:1])
        pizeros_tau2  = ak.firsts(pizeros[:,1:2])

        #from IPython import embed; embed()
        extra_feats = []
        extra_keys   = []
        _feats = ["pt", "eta", "phi", "mass"]
        for feat in _feats:
            #print(f" --- {feat}")
            for i in range(3):
                #print(f" ------ {i}")
                pi_temp_tau1  = pions_tau1[feat][:,i:i+1]
                #print(pi_temp_tau1)
                pi_temp_tau2  = pions_tau2[feat][:,i:i+1]
                #print(pi_temp_tau2)
                pi0_temp_tau1 = ak.from_regular(pizeros_tau1[feat][:,i:i+1])
                #print(pi0_temp_tau1)
                pi0_temp_tau2 = ak.from_regular(pizeros_tau2[feat][:,i:i+1])
                #print(pi0_temp_tau2)
                #from IPython import embed; embed()

                temp_key_list  = [f"{feat}_tau1pi_{i+1}", f"{feat}_tau2pi_{i+1}", f"{feat}_tau1pi0_{i+1}", f"{feat}_tau2pi0_{i+1}"]
                temp_feat_list = ak.concatenate([ak.fill_none(ak.firsts(pi_temp_tau1),  0.0)[:,None],
                                                 ak.fill_none(ak.firsts(pi_temp_tau2),  0.0)[:,None],
                                                 ak.fill_none(ak.firsts(pi0_temp_tau1), 0.0)[:,None],
                                                 ak.fill_none(ak.firsts(pi0_temp_tau2), 0.0)[:,None]], axis=1)

                extra_feats.append(temp_feat_list)
                extra_keys = extra_keys + temp_key_list

        """
        extra_feats = ak.concatenate([pions_tau1.pt, pions_tau1.eta, pions_tau1.phi, pions_tau1.mass,
                                      pions_tau2.pt, pions_tau2.eta, pions_tau2.phi, pions_tau2.mass,
                                      pizeros_tau1.pt, pizeros_tau1.eta, pizeros_tau1.phi, pizeros_tau1.mass,
                                      pizeros_tau2.pt, pizeros_tau2.eta, pizeros_tau2.phi, pizeros_tau2.mass], axis=1)
        extra_keys = ["pt_pi_1", "eta_pi_1", "phi_pi_1", "mass_pi_1",
                      "pt_pi_2", "eta_pi_2", "phi_pi_2", "mass_pi_2",
                      "pt_pi0_1", "eta_pi0_1", "phi_pi0_1", "mass_pi0_1",
                      "pt_pi0_2", "eta_pi0_2", "phi_pi0_2", "mass_pi0_2"]
        """
        extra_feats = ak.concatenate(extra_feats, axis=1)
        
        phi_cp_gen     = -99 * ak.ones_like(events.event)[:,None]
        phi_cp_gen_det = -99 * ak.ones_like(events.event)[:,None]

        h1x_gen = -99 * ak.ones_like(events.event)[:,None]
        h1y_gen = -99 * ak.ones_like(events.event)[:,None]
        h1z_gen = -99 * ak.ones_like(events.event)[:,None]
        h2x_gen = -99 * ak.ones_like(events.event)[:,None]
        h2y_gen = -99 * ak.ones_like(events.event)[:,None]
        h2z_gen = -99 * ak.ones_like(events.event)[:,None]

        k1x_gen = -99 * ak.ones_like(events.event)[:,None]
        k1y_gen = -99 * ak.ones_like(events.event)[:,None]
        k1z_gen = -99 * ak.ones_like(events.event)[:,None]
        k2x_gen = -99 * ak.ones_like(events.event)[:,None]
        k2y_gen = -99 * ak.ones_like(events.event)[:,None]
        k2z_gen = -99 * ak.ones_like(events.event)[:,None]
        
        k1rx_gen = -99 * ak.ones_like(events.event)[:,None]
        k1ry_gen = -99 * ak.ones_like(events.event)[:,None]
        k1rz_gen = -99 * ak.ones_like(events.event)[:,None]
        k2rx_gen = -99 * ak.ones_like(events.event)[:,None]
        k2ry_gen = -99 * ak.ones_like(events.event)[:,None]
        k2rz_gen = -99 * ak.ones_like(events.event)[:,None]


        if self.isForTrain:
            #from IPython import embed; embed()
            phicp_obj_gen = PhiCPComp(cat="rhorho",
                                      taum=self.gentaus[:,0:1],
                                      taup=self.gentaus[:,1:2],
                                      taum_decay=self.gentauprods[:,0:1],
                                      taup_decay=self.gentauprods[:,1:2])
            phi_cp_gen, hk_dict = phicp_obj_gen.comp_phiCP()
            phi_cp_gen = ak.firsts(phi_cp_gen, axis=1)
            
            h1x_gen = ak.firsts(hk_dict["h1_unit"].x, axis=1)
            h1y_gen = ak.firsts(hk_dict["h1_unit"].y, axis=1)
            h1z_gen = ak.firsts(hk_dict["h1_unit"].z, axis=1)
            h2x_gen = ak.firsts(hk_dict["h2_unit"].x, axis=1)
            h2y_gen = ak.firsts(hk_dict["h2_unit"].y, axis=1)
            h2z_gen = ak.firsts(hk_dict["h2_unit"].z, axis=1)
            k1x_gen = ak.firsts(hk_dict["k1_unit"].x, axis=1)
            k1y_gen = ak.firsts(hk_dict["k1_unit"].y, axis=1)
            k1z_gen = ak.firsts(hk_dict["k1_unit"].z, axis=1)
            k2x_gen = ak.firsts(hk_dict["k2_unit"].x, axis=1)
            k2y_gen = ak.firsts(hk_dict["k2_unit"].y, axis=1)
            k2z_gen = ak.firsts(hk_dict["k2_unit"].z, axis=1)
            k1rx_gen = ak.firsts(hk_dict["k1_raw"].x, axis=1)
            k1ry_gen = ak.firsts(hk_dict["k1_raw"].y, axis=1)
            k1rz_gen = ak.firsts(hk_dict["k1_raw"].z, axis=1)
            k2rx_gen = ak.firsts(hk_dict["k2_raw"].x, axis=1)
            k2ry_gen = ak.firsts(hk_dict["k2_raw"].y, axis=1)
            k2rz_gen = ak.firsts(hk_dict["k2_raw"].z, axis=1)


            phicp_gen_det_obj = PhiCPComp(cat="rhorho",
                                          taum=self.gentaus[:,0:1],
                                          taup=self.gentaus[:,1:2],
                                          taum_decay=_tauprods[:,0:1],
                                          taup_decay=_tauprods[:,1:2])
            phi_cp_gen_det, _ = phicp_gen_det_obj.comp_phiCP()
            phi_cp_gen_det    = ak.firsts(phi_cp_gen_det, axis=1)
            
            
            print(f"phi_cp_gen      : {phi_cp_gen}")
            print(f"phi_cp (gen_det): {phi_cp_gen_det}")
            
            
            new_extra_feats = ak.concatenate([phi_cp_gen, phi_cp_gen_det,
                                              h1x_gen, h1y_gen, h1z_gen, h2x_gen, h2y_gen, h2z_gen,
                                              k1x_gen, k1y_gen, k1z_gen, k2x_gen, k2y_gen, k2z_gen,
                                              k1rx_gen, k1ry_gen, k1rz_gen, k2rx_gen, k2ry_gen, k2rz_gen], axis=1)
            new_extra_keys = ["phicp", "phi_cp_det", "h1x", "h1y", "h1z", "h2x", "h2y", "h2z", "k1x", "k1y", "k1z",
                              "k2x", "k2y", "k2z", "k1rx", "k1ry", "k1rz", "k2rx", "k2ry", "k2rz"]


            extra_feats = ak.concatenate([extra_feats, new_extra_feats], axis=1)
            extra_keys = extra_keys + new_extra_keys
            

        np_extra_feats = ak.to_numpy(extra_feats)

        print(f"extra    : {np_extra_feats.shape}, {len(extra_keys)}")
        return np_extra_feats, extra_keys




    def extraction(self):
        np_manual_node_feats, manual_node_keys = self.getmanuals()
        np_tau_node_feats, tau_node_keys = self.getnodes("tau")
        np_jet_node_feats, jet_node_keys = self.getnodes("jet")
        np_global_feats, global_keys = self.getglobals(self.global_var_to_keep())
        np_target_feats, target_keys = self.gettargets()


        print("details: ===>")
        print(f"manual   : {np_manual_node_feats.shape}, {len(manual_node_keys)}")
        print(f"node_tau : {np_tau_node_feats.shape}, {len(tau_node_keys)}")
        print(f"node_jet : {np_jet_node_feats.shape}, {len(jet_node_keys)}")
        print(f"global   : {np_global_feats.shape}, {len(global_keys)}")
        print(f"target   : {np_target_feats.shape}, {len(target_keys)}")

        all_feats = np.concatenate((np_manual_node_feats, 
                                    np_tau_node_feats,
                                    np_jet_node_feats,
                                    np_global_feats,
                                    np_target_feats), axis=1)
        all_keys = manual_node_keys + tau_node_keys + jet_node_keys + global_keys + target_keys

        return all_feats, all_keys
