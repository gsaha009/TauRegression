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
from typing import Optional
from PhiCPComp import PhiCPComp



def invariant_mass(events: ak.Array) -> ak.Array:
    empty_events = ak.zeros_like(events, dtype=np.uint16)[:, 0:0]
    where = ak.num(events, axis=1) == 2
    events_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((1 * events_2[:, :1] + 1 * events_2[:, 1:2]).mass), 0)
    return mass

def getconfig(filepath):
    with open(filepath,'r') as conf:
        config = yaml.safe_load(conf)
    return config

def getlistofinputs(paths):
    files = []
    for path in paths:
        temp = glob.glob(f"{path}/*.root")
        files = files + temp
    print(f"Files: {files}")
    return files

def muon_selection(events: ak.Array) -> ak.Array:
    muos = events.Muon
    sorted_indices = ak.argsort(muos.pt, axis=-1, ascending=False)
    muo_mask = (
        (muos.pt > 20) &
        (np.abs(muos.eta) < 2.4) &
        (abs(muos.dxy) < 0.045) &
        (abs(muos.dz) < 0.2) &
        (muos.mediumId == 1)
    )

    sel_muo_indices = sorted_indices[muo_mask[sorted_indices]]
    #print("muons selected")
    return sel_muo_indices


def electron_selection(events: ak.Array, mu_idxs=None) -> ak.Array:
    eles = events.Electron
    sorted_indices = ak.argsort(eles.pt, axis=-1, ascending=False)
    ele_mask = (
        (eles.pt > 25) &
        (np.abs(eles.eta) < 2.5) &
        (abs(eles.dxy) < 0.045) &
        (abs(eles.dz) < 0.2) &
        (eles.mvaIso_WP80 == 1)
        #(eles.mvaFall17V2Iso_WP80 == 1)
    )
    if mu_idxs is not None:
        ele_mask = ele_mask & ak.all(eles.metric_table(events.Muon[mu_idxs]) > 0.2, axis=2)
    
    sel_ele_indices = sorted_indices[ele_mask[sorted_indices]]
    #print("electrons selected")
    return sel_ele_indices


def tau_selection(events: ak.Array, ele_idxs=None, mu_idxs=None) -> ak.Array:
    taus = events.Tau
    sorted_indices = ak.argsort(taus.pt, axis=-1, ascending=False)
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
        & (taus.idDeepTau2017v2p1VSjet >= 4) #4
        & (taus.idDeepTau2017v2p1VSe >= 2)
        & (taus.idDeepTau2017v2p1VSmu >= 1)
    )
    if ele_idxs is not None:
        tau_mask = tau_mask & ak.all(taus.metric_table(events.Electron[ele_idxs]) > 0.2, axis=2)
    if mu_idxs is not None:
        tau_mask = tau_mask & ak.all(taus.metric_table(events.Muon[mu_idxs]) > 0.2, axis=2)

    sel_tau_indices = sorted_indices[tau_mask[sorted_indices]]
    #print("taus selected")
    return sel_tau_indices


def jet_selection(events: ak.Array, ele_idxs=None, mu_idxs=None, tau_idxs=None) -> ak.Array:
    jets = events.Jet
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
        jet_mask = jet_mask & ak.all(jets.metric_table(events.Electron[ele_idxs]) > 0.2, axis=2)
    if mu_idxs is not None:
        jet_mask = jet_mask & ak.all(jets.metric_table(events.Muon[mu_idxs]) > 0.2, axis=2)
    if tau_idxs is not None:
        jet_mask = jet_mask & ak.all(jets.metric_table(events.Tau[tau_idxs]) > 0.2, axis=2)

    sel_jet_indices = sorted_indices[jet_mask[sorted_indices]]
    #print("jets selected")
    return sel_jet_indices


def gentauandprods_selection(events: ak.Array):
    # masks to select gen tau+ and tau-
    is_taum = ((events.GenPart.pdgId == 15) 
               & (events.GenPart.hasFlags(["isPrompt","isLastCopy"]))
               & (events.GenPart.status == 2)
               & (events.GenPart.pt >= 10)
               & (np.abs(events.GenPart.eta) <= 2.3))
    is_taup = ((events.GenPart.pdgId == -15) 
               & (events.GenPart.hasFlags(["isPrompt","isLastCopy"])) 
               & (events.GenPart.status == 2)
               & (events.GenPart.pt >= 10)
               & (np.abs(events.GenPart.eta) <= 2.3))
    
    # get gen tau+ and tau- objects from events GenParts
    taum = events.GenPart[is_taum]
    taup = events.GenPart[is_taup]

    # make sure event has at least one tau+ and tau-
    has_taum = ak.sum(is_taum, axis=1) > 0
    has_taup = ak.sum(is_taup, axis=1) > 0
    evt_mask_1 = has_taum & has_taup
    
    events = events[evt_mask_1]
    taum = taum[evt_mask_1]
    taup = taup[evt_mask_1]
        
    taum_momidx    = taum.distinctParent.genPartIdxMother
    #taum_mom       = events.GenPart[taum_momidx]
    is_taum_from_h = (events.GenPart[taum_momidx].pdgId == 25) & (events.GenPart[taum_momidx].status == 22)
    # get tau- mother indices
    taum_momidx = taum_momidx[is_taum_from_h]
    #taum_mom    = events.GenPart[taum_momidx]

    taup_momidx = taup.distinctParent.genPartIdxMother
    #taup_mom    = events.GenPart[taup_momidx]
    is_taup_from_h = (events.GenPart[taup_momidx].pdgId == 25) & (events.GenPart[taup_momidx].status == 22)
    # get tau+ mother indices
    taup_momidx = taup_momidx[is_taup_from_h]
    #taup_mom    = events.GenPart[taup_momidx]

    has_taum_from_h = ak.sum(is_taum_from_h, axis=1) > 0
    has_taup_from_h = ak.sum(is_taup_from_h, axis=1) > 0

    evt_mask_2 = has_taum_from_h & has_taup_from_h
    events = events[evt_mask_2]
    taum = taum[evt_mask_2]
    taup = taup[evt_mask_2]
    taum_momidx = taum_momidx[evt_mask_2]
    taup_momidx = taup_momidx[evt_mask_2]


    # get tau products
    taum_children = ak.firsts(taum.distinctChildren)
    taup_children = ak.firsts(taup.distinctChildren)

    evt_mask_3 = (ak.num(taum_children.pdgId, axis=1) > 0) & (ak.num(taup_children.pdgId, axis=1) > 0)
    events = events[evt_mask_3]
    taum = taum[evt_mask_3]
    taup = taup[evt_mask_3]
    taum_momidx = taum_momidx[evt_mask_3]
    taup_momidx = taup_momidx[evt_mask_3]
    taum_children = taum_children[evt_mask_3]
    taup_children = taup_children[evt_mask_3]


    #event_level_mask = ((ak.num(taum_momidx, axis=1)   == 1) 
    #                    & (ak.num(taup_momidx, axis=1) == 1)
    #                    & (ak.num(taum.pdgId, axis=1)  == 1)
    #                    & (ak.num(taup.pdgId, axis=1)  == 1)
    #                    & (ak.num(taum_children.pdgId, axis=1) > 0)
    #                    & (ak.num(taup_children.pdgId, axis=1) > 0))

    #events = events[event_level_mask]
    #taum = taum[event_level_mask]
    #taup = taup[event_level_mask]
    #taum_momidx = taum_momidx[event_level_mask]
    #taup_momidx = taup_momidx[event_level_mask]

    #same_mom = taum_momidx == taup_momidx
    #evt_level_mask4 = ak.sum(same_mom, axis=1) == 1
    evt_level_mask4 = ((ak.num(taum.pdgId, axis=1) == 1) 
                       & (ak.num(taup.pdgId, axis=1) == 1)
                       & (ak.num(taum_momidx, axis=1) == 1)
                       & (ak.num(taup_momidx, axis=1) == 1))

    events = events[evt_level_mask4]
    taum = taum[evt_level_mask4]
    taup = taup[evt_level_mask4]
    taum_children = taum_children[evt_level_mask4]
    taup_children = taup_children[evt_level_mask4]

    #from IPython import embed; embed()


    #taum = taum[same_mom]
    #taup = taup[same_mom]
    #taum_children = ak.firsts(taum.distinctChildren)
    #taup_children = ak.firsts(taup.distinctChildren)

    #event_level_mask2 = ak.sum(same_mom, axis=1) == 1

    #events = events[event_level_mask2]
    #taum = taum[event_level_mask2]
    #taup = taup[event_level_mask2]
    #taum_children = taum_children[event_level_mask2]
    #taup_children = taup_children[event_level_mask2]


    return events, taum, taup, taum_children, taup_children
    


def gentaunu_selection(events: ak.Array, tau_idxs=None) -> ak.Array:
    gens = events.GenPart
    local_indices = ak.local_index(gens.pdgId, axis=-1)

    istaunu = np.abs(gens.pdgId) == 16
    isdecayfromtau = gens.hasFlags(["isTauDecayProduct",
                                    "isPromptTauDecayProduct",
                                    "isDirectTauDecayProduct",
                                    "isDirectPromptTauDecayProduct"])
    hastauasmom = np.abs(gens.parent.pdgId) == 15
    gen_mask = istaunu & isdecayfromtau & hastauasmom

    if tau_idxs is not None:
        gen_mask = gen_mask & ak.any(gens.metric_table(events.Tau[tau_idxs]) < 0.4, axis=2)

    gen_sel_indices = local_indices[gen_mask]

    return gen_sel_indices



def gentau_selection(events: ak.Array, tau_idxs=None) -> ak.Array:
    gens = events.GenPart
    local_indices = ak.local_index(gens.pdgId, axis=-1)
    
    #print(f"Gens: {gens.pdgId[:1]}")
    #print(f"Mother: {gens.parent.pdgId[:1]}")
    #from IPython import embed; embed()

    istau = np.abs(gens.pdgId) == 15
    isfromZ = gens.hasFlags(["isFirstCopy"])
    hasZasmom = gens.parent.pdgId == 23
    gen_mask = istau & isfromZ & hasZasmom
    
    if tau_idxs is not None:
        gen_mask = gen_mask & ak.any(gens.metric_table(events.Tau[tau_idxs]) < 0.4, axis=2)

    gen_sel_indices = local_indices[gen_mask]
    #from IPython import embed; embed()
    return gen_sel_indices



def get_MT(objs: ak.Array, met: ak.Array) -> ak.Array:
    dphi = (1*objs).delta_phi(1*met)
    dphi = ak.fill_none(ak.firsts(dphi, axis=1), 0.0)
    #print(f"dphi: {dphi}")
    mt = np.sqrt(2*met.pt*objs.pt*(1 - np.cos(dphi)))
    #print(f"mt: {mt}")
    return mt
    #return ak.fill_none(ak.firsts(mt, axis=1), 0.0)

    
def get_total_MT(obj1: ak.Array, obj2: ak.Array, met: ak.Array) -> ak.Array:
    return np.sqrt((get_MT(obj1, met))**2 + get_MT(obj2, met)**2 + get_MT(obj1, obj2)**2)


scalefeat_flat = lambda arr: (arr - ak.mean(arr))/ak.std(arr)
def scalefeat_nest(arr: ak.Array, ax: int)->ak.Array:
    max_num = ak.max(ak.count(arr, axis=ax))
    scaled_arr = None
    for i in range(max_num):
        #print(i)
        temp = arr[:,i:i+1]
        temp_mean = ak.mean(temp)
        std_temp = ak.std(temp)
        temp_scaled = temp - temp_mean if std_temp == 0.0 else (temp - temp_mean)/std_temp
        #print(f"scaled: {temp_scaled}")
        scaled_arr = temp_scaled if i == 0 else ak.concatenate([scaled_arr, temp_scaled], axis=ax)
    return scaled_arr
    
def filldict(dict={}, colname="", array=None, iteridx=0):
    if iteridx == 0:
        dict[colname] = ak.to_numpy(array).reshape(-1)
    else:
        np.append(dict[colname], ak.to_numpy(array).reshape(-1))


def getP4(arr: ak.Array) -> ak.Array:
    TetraVec = ak.zip(
        {"pt": arr.pt, 
         "eta": arr.eta, 
         "phi": arr.phi, 
         "mass": arr.mass,
         "pdgId": arr.pdgId if "pdgId" in arr.fields else -99.9*ak.ones_like(arr.pt),
         "tauIdx": arr.pdgId if "tauIdx" in arr.fields else -99.9*ak.ones_like(arr.pt)},
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior)
    return TetraVec

# ----------------------------------------------------------------------------------------------------- #
#                                            Node features                                              #
# ----------------------------------------------------------------------------------------------------- #
def tonumpy(arr, maxidx, iscat, isnorm, feat, tag):
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
        elif isnorm and np.sum(temp) > 0.0:
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


def get_feats_per_obj(col, featlist, maxidx, _tag, isnorm):
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
        var_keys, np_var = tonumpy(var, maxidx, iscat, isnorm, feat, _tag) # feat: px, tag: tau
        
        if i == 0:
            node_np = np_var
            keys_np = var_keys
        else:
            node_np = np.concatenate((node_np, np_var), axis=1)
            keys_np = keys_np + var_keys

    return keys_np, node_np

def getnodes(objs, featlist, maxidx, tag, isnorm):
    #keys_tau, node_feat_tau = get_feats_per_obj(taus, featlist, 2, "tau")
    #keys_jet, node_feat_jet = get_feats_per_obj(jets, featlist, 10, "jet")

    #keys  = keys_tau + keys_jet
    #feats = np.concatenate((node_feat_tau, node_feat_jet), axis=1)

    keys, feats = get_feats_per_obj(objs, featlist, maxidx, tag, isnorm)
    print(f"Total node feats: {len(keys)}")
    print(f"Node feat shape: {feats.shape}")

    return feats, keys

def getmanuals(taus, tau1prods, tau2prods, met, isnorm):
    tau1prods_p4 = getP4(tau1prods)
    tau2prods_p4 = getP4(tau2prods)
    taus_p4     = getP4(taus)
    
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
    if isnorm:
        concat_array = (concat_array - np.mean(concat_array, axis=0))/np.std(concat_array, axis=0)

    print(f"Total manual feats: {len(key_list)}")
    print(f"Node manual shape: {concat_array.shape}")

    return concat_array, key_list


# ----------------------------------------------------------------------------------------------------- #
#                                          Global features                                              #
# ----------------------------------------------------------------------------------------------------- #
def global_var_to_keep(events: ak.Array, taus: ak.Array):
    globalDict = {
        "pt_MET": {"coll": events.MET, "fname": "pt"},
        "phi_MET": {"coll": events.MET, "fname": "phi"},
        "covXX_MET": {"coll": events.MET, "fname": "covXX"},
        "covXY_MET": {"coll": events.MET, "fname": "covXY"},
        "covYY_MET": {"coll": events.MET, "fname": "covYY"},
        "sumEt_MET": {"coll": events.MET, "fname": "sumEt"},
        "significance_MET": {"coll": events.MET, "fname": "significance"},        
        "nJet": {"coll": events.Jet, "fname": "num"},
        "nPV" :  {"coll": events.PV, "fname": "npvsGood"},
        "HT"  : {"coll": events.Jet.pt, "fname": "sum"},
        "MT_tau1_met" : {"coll": [taus[:,:1], events.MET], "fname": lambda x, y: get_MT(x,y)},
        "MT_tau2_met" : {"coll": [taus[:,1:2], events.MET], "fname": lambda x, y: get_MT(x,y)},
        "MT_tau1_tau2": {"coll": [taus[:,:1], taus[:,1:2]], "fname": lambda x, y: get_MT(x,y)},
        "MT_total": {"coll": [taus[:,:1], taus[:,1:2], events.MET], "fname": lambda x, y, z: get_total_MT(x,y,z)},
    }
    return globalDict


def getglobals(vardict, isnorm):
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

    if isnorm:
        global_np = (global_np - np.mean(global_np, axis=0))/np.std(global_np, axis=0)

    #data  = pd.DataFrame(global_np, columns=global_feats)
    #return data
    print(f"Total global feats: {len(global_feats)}")
    print(f"Global feat shape: {global_np.shape}")
    return global_np, global_feats


# ----------------------------------------------------------------------------------------------------- #
#                                            get targets                                                #
# ----------------------------------------------------------------------------------------------------- #
def gettargets(gentaus, gentaunus):
    gentau_1_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].px, axis=1), 0.0))[:,None]
    gentau_1_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].py, axis=1), 0.0))[:,None]
    gentau_1_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,:1].pz, axis=1), 0.0))[:,None]
    gentau_2_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].px, axis=1), 0.0))[:,None]
    gentau_2_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].py, axis=1), 0.0))[:,None]
    gentau_2_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaus[:,1:2].pz, axis=1), 0.0))[:,None]

    gentaunu_1_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].px, axis=1), 0.0))[:,None]
    gentaunu_1_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].py, axis=1), 0.0))[:,None]
    gentaunu_1_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,:1].pz, axis=1), 0.0))[:,None]
    gentaunu_2_px = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].px, axis=1), 0.0))[:,None]
    gentaunu_2_py = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].py, axis=1), 0.0))[:,None]
    gentaunu_2_pz = ak.to_numpy(ak.fill_none(ak.firsts(gentaunus[:,1:2].pz, axis=1), 0.0))[:,None]
 
    np_target_feats = np.concatenate((gentau_1_px, gentau_1_py, gentau_1_pz, gentau_2_px, gentau_2_py, gentau_2_pz,
                                      gentaunu_1_px, gentaunu_1_py, gentaunu_1_pz, gentaunu_2_px, gentaunu_2_py, gentaunu_2_pz), axis=1)
    target_keys = ["px_gentau_1", "py_gentau_1", "pz_gentau_1", 
                   "px_gentau_2", "py_gentau_2", "pz_gentau_2",
                   "px_gentaunu_1", "py_gentaunu_1", "pz_gentaunu_1", 
                   "px_gentaunu_2", "py_gentaunu_2", "pz_gentaunu_2"]

    print(f"Total target feats: {len(target_keys)}")
    print(f"Target feat shape: {np_target_feats.shape}")

    return np_target_feats, target_keys


def findDecayMode(Nc, Np): 
    return 5 * (Nc - 1) + Np


def genDM(prod: ak.Array):
    pids = prod.pdgId
    is_charged = ((np.abs(pids) == 211) 
                  | (np.abs(pids) == 321))
    is_neutral = ((pids == 111) | (pids == 311))
    
    Nc = ak.sum(is_charged, axis=-1)
    Np = ak.sum(is_neutral, axis=-1)

    dm = findDecayMode(Nc, Np)
    
    return dm


# ----------------------------------------------------------------------------------------------------- #
#                                            main script                                                #
# ----------------------------------------------------------------------------------------------------- #
def get_events_dict(_events:ak.Array, norm: bool, dm: str):
    print(f"Decay Mode: {dm}")
    events, taum, taup, taum_children, taup_children = gentauandprods_selection(_events)

    muo_sel_idx = muon_selection(events)
    ele_sel_idx = electron_selection(events, mu_idxs=muo_sel_idx)
    tau_sel_idx = tau_selection(events, ele_idxs=ele_sel_idx, mu_idxs=muo_sel_idx)
    jet_sel_idx = jet_selection(events, ele_idxs=ele_sel_idx, mu_idxs=muo_sel_idx, tau_idxs=tau_sel_idx)
    #gentaunu_sel_idx = gentaunu_selection(events, tau_idxs=tau_sel_idx)
    #gentau_sel_idx = gentau_selection(events, tau_idxs=tau_sel_idx)

    #print(tau_sel_idx)
        
    is_tauhtauh = (
        (ak.num(tau_sel_idx, axis=1) == 2)
        & (ak.num(ele_sel_idx, axis=1) == 0)
        & (ak.num(muo_sel_idx, axis=1) == 0)
        & (ak.num(taum, axis=1) == 1)
        & (ak.num(taup, axis=1) == 1)
        & (ak.num(taum_children, axis=1) > 0)
        & (ak.num(taup_children, axis=1) > 0)
    )

    is_os = ak.sum(events.Tau[tau_sel_idx].charge, axis=1) == 0
    #has_two_gentaunu = ak.num(gentaunu_sel_idx, axis=-1) >= 2
    #has_two_gentau   = ak.num(gentau_sel_idx, axis=-1) >= 2
    
    where = is_tauhtauh & is_os

    #from IPython import embed; embed()
    
    events = events[where]
    tau_sel_idx = tau_sel_idx[where]
    jet_sel_idx = jet_sel_idx[where]
    #gentaunu_sel_idx = ak.where(where, gentaunu_sel_idx, empty_indices)
    #gentau_sel_idx = ak.where(where, gentau_sel_idx, empty_indices)
    
    jets = events.Jet[jet_sel_idx]
    taus = events.Tau[tau_sel_idx]
    #gentaunus = events.GenPart[gentaunu_sel_idx]
    #gentaus = events.GenPart[gentau_sel_idx]
    tauprods = events.TauProd
    # gentaus
    gentaum = taum[where]
    gentaup = taup[where]
    gentaum_decay = taum_children[where]
    gentaup_decay = taup_children[where]

    gentaus = ak.concatenate([gentaum, gentaup], axis=1)


    _taus = gentaus.nearest(taus, threshold=0.4)

    mask = (ak.num(_taus, axis=1) == 2) & (ak.sum(taus.charge, axis=1) == 0)

    events = events[mask]
    taus = taus[mask]
    tauprods = tauprods[mask]
    jets = jets[mask]
    gentaum = gentaum[mask]
    gentaup = gentaup[mask]
    gentaum_decay = gentaum_decay[mask]
    gentaup_decay = gentaup_decay[mask]

    gentaum_decay_nu = gentaum_decay[np.abs(gentaum_decay.pdgId) == 16]
    gentaup_decay_nu = gentaup_decay[np.abs(gentaup_decay.pdgId) == 16]
    
    gentaunus = ak.concatenate([gentaum_decay_nu, gentaup_decay_nu], axis=1)


    assert ak.min(ak.num(gentaunus.pdgId, axis=1)) == 2

    gentaus     = ak.concatenate([gentaum, gentaup], axis=1)
    gentauprods = ak.concatenate([gentaum_decay[:,None], gentaup_decay[:,None]], axis=1)
    #print(f"Fields: {gentaunus.fields}, {gentaus.fields}")
    #print(f"Gen_nu mass: {ak.max(gentaunus.mass)}")
    #print(f"Gen_tau mass: {ak.max(gentaus.mass)}")
    
    gentaunus["px"] = gentaunus.px
    gentaunus["py"] = gentaunus.py
    gentaunus["pz"] = gentaunus.pz
    #gentaunus["energy"] = gentaunus.E
    #gentaunus["mass"] = (1 * gentaunus).mass
    
    gentaus["px"] = gentaus.px
    gentaus["py"] = gentaus.py
    gentaus["pz"] = gentaus.pz
    #gentaus["energy"] = gentaus.E
    #gentaus["mass"] = (1 * gentaus).mass
    gentaus["DM"] = genDM(gentauprods)

    
    # get tau prods for taus
    tau1_idx = tau_sel_idx[:,:1]
    tau1_idx_brdcst, tauprod_tauIdx = ak.broadcast_arrays(ak.firsts(tau1_idx,axis=1), tauprods.tauIdx)
    tau1prod_mask = tauprod_tauIdx == tau1_idx_brdcst
    tau2_idx = tau_sel_idx[:,1:2]
    tau2_idx_brdcst, _ = ak.broadcast_arrays(ak.firsts(tau2_idx,axis=1), tauprods.tauIdx)
    tau2prod_mask = tauprod_tauIdx == tau2_idx_brdcst
    tau1prods = tauprods[tau1prod_mask]
    tau2prods = tauprods[tau2prod_mask]

    #print(tau1_prods.tauIdx)
    #print(tau2_prods.tauIdx)
    tauprods_concat = ak.concatenate([tau1prods[:,None], tau2prods[:,None]], axis=1)
    """
    final_is_os = ak.sum(taus.charge, axis=1) == 0
    events      = events[final_is_os] 
    taus        = taus[final_is_os]
    tau1prods   = tau1prods[final_is_os]
    tau2prods   = tau2prods[final_is_os] 
    tauprods_concat = ak.concatenate([tau1prods[:,None], tau2prods[:,None]], axis=1)
    jets        = jets[final_is_os] 
    gentaus     = gentaus[final_is_os]
    gentaunus   = gentaunus[final_is_os]
    gentauprods = gentauprods[final_is_os]
    """
    # --------------------------------------------------------------- #
    # Events has two gen taus and detector level taus
    # calculate from gen-level only:
    # -- get decay modes from gen - level
    # -- compare with reconstructed level DM
    #print(f"charge: {ak.to_list(taus.charge)}")

    print(f"taus pt      : {taus.pt}")
    print(f"gentaus pt   : {gentaus.pt}")
    print(f"tau prods    : {tauprods_concat.pdgId}")
    print(f"gentau prods : {gentauprods.pdgId}")
    print(f"gentau DM    : {gentaus.DM}")


    # ----------
    # zero prong only
    # ----------
    mask = events.event >= 0
    catstr = ""
    if dm == "00":
        zero_prong = (gentaus.DM == 0)
        mask = ak.sum(zero_prong, axis=1) == 2
        catstr = "pipi"
    elif dm == "10":
        one_zero_prong = (((gentaus.DM[:,:1] == 0) & (gentaus.DM[:,1:2] == 1)) 
                          | ((gentaus.DM[:,:1] == 1) & (gentaus.DM[:,1:2] == 0)))
        mask = ak.sum(one_zero_prong, axis=1) == 2
        catstr = "pirho"
    elif dm == "11":
        one_prong = (gentaus.DM == 1)
        mask = ak.sum(one_prong, axis=1) == 2
        catstr = "rhorho"
    elif dm == "1010":
        one_prong = (gentaus.DM == 10)
        mask = ak.sum(one_prong, axis=1) == 2
        catstr = "a1a1"

    else:
        print("WARNING: inclusive DM")
    
    events      = events[mask] 
    taus        = taus[mask]
    tau1prods   = tau1prods[mask]
    tau2prods   = tau2prods[mask] 
    tauprods_concat = ak.concatenate([tau1prods[:,None], tau2prods[:,None]], axis=1)
    jets        = jets[mask] 
    gentaus     = gentaus[mask]
    gentaunus   = gentaunus[mask]
    gentauprods = gentauprods[mask]
    


    # -------------------------------------------------------------------------------------------------------------------------- #
    # Everything is available now. 
    # -------------------------------------------------------------------------------------------------------------------------- #

    
    
    print(f"adskjcnkj TAU FIELDS: {gentaus.fields}")
    
    from IPython import embed; embed()
    1/0

    phi_cp = -99 * ak.ones_like(events.event)[:,None]

    h1x = -99 * ak.ones_like(events.event)[:,None]
    h1y = -99 * ak.ones_like(events.event)[:,None]
    h1z = -99 * ak.ones_like(events.event)[:,None]
    h2x = -99 * ak.ones_like(events.event)[:,None]
    h2y = -99 * ak.ones_like(events.event)[:,None]
    h2z = -99 * ak.ones_like(events.event)[:,None]

    k1x = -99 * ak.ones_like(events.event)[:,None]
    k1y = -99 * ak.ones_like(events.event)[:,None]
    k1z = -99 * ak.ones_like(events.event)[:,None]
    k2x = -99 * ak.ones_like(events.event)[:,None]
    k2y = -99 * ak.ones_like(events.event)[:,None]
    k2z = -99 * ak.ones_like(events.event)[:,None]

    k1rx = -99 * ak.ones_like(events.event)[:,None]
    k1ry = -99 * ak.ones_like(events.event)[:,None]
    k1rz = -99 * ak.ones_like(events.event)[:,None]
    k2rx = -99 * ak.ones_like(events.event)[:,None]
    k2ry = -99 * ak.ones_like(events.event)[:,None]
    k2rz = -99 * ak.ones_like(events.event)[:,None]

    hasdm = (dm == "00") | (dm == "10") | (dm == "11") | (dm == "1010")
    if hasdm:
        phicp_obj = PhiCPComp(cat=catstr,
                              taum=gentaus[:,0:1],
                              taup=gentaus[:,1:2],
                              taum_decay=gentauprods[:,0:1],
                              taup_decay=gentauprods[:,1:2])
        phi_cp, hk_dict = phicp_obj.comp_phiCP()
        phi_cp = ak.firsts(phi_cp, axis=1)
        #print(k_dict["k1_unit"].x, k_dict["k1_unit"].y, k_dict["k1_unit"].z)
        #print(k_dict["k2_unit"].x, k_dict["k2_unit"].y, k_dict["k2_unit"].z)
        h1x = ak.firsts(hk_dict["h1_unit"].x, axis=1)
        h1y = ak.firsts(hk_dict["h1_unit"].y, axis=1)
        h1z = ak.firsts(hk_dict["h1_unit"].z, axis=1)
        h2x = ak.firsts(hk_dict["h2_unit"].x, axis=1)
        h2y = ak.firsts(hk_dict["h2_unit"].y, axis=1)
        h2z = ak.firsts(hk_dict["h2_unit"].z, axis=1)
        k1x = ak.firsts(hk_dict["k1_unit"].x, axis=1)
        k1y = ak.firsts(hk_dict["k1_unit"].y, axis=1)
        k1z = ak.firsts(hk_dict["k1_unit"].z, axis=1)
        k2x = ak.firsts(hk_dict["k2_unit"].x, axis=1)
        k2y = ak.firsts(hk_dict["k2_unit"].y, axis=1)
        k2z = ak.firsts(hk_dict["k2_unit"].z, axis=1)
        k1rx = ak.firsts(hk_dict["k1_raw"].x, axis=1)
        k1ry = ak.firsts(hk_dict["k1_raw"].y, axis=1)
        k1rz = ak.firsts(hk_dict["k1_raw"].z, axis=1)
        k2rx = ak.firsts(hk_dict["k2_raw"].x, axis=1)
        k2ry = ak.firsts(hk_dict["k2_raw"].y, axis=1)
        k2rz = ak.firsts(hk_dict["k2_raw"].z, axis=1)

    print(f"phi_cp: {phi_cp}")
    print(f"h1x: {h1x}")
    print(f"k1x: {k1x}")
    #from IPython import embed; embed()
    """
    mask = ak.fill_none(ak.num(phi_cp, axis=1),0) == 1
    
    events      = events[mask] 
    taus        = taus[mask]
    tau1prods   = tau1prods[mask]
    tau2prods   = tau2prods[mask] 
    tauprods_concat = ak.concatenate([tau1prods[:,None], tau2prods[:,None]], axis=1)
    jets        = jets[mask] 
    gentaus     = gentaus[mask]
    gentaunus   = gentaunus[mask]
    gentauprods = gentauprods[mask]
    phi_cp      = phi_cp[mask]
    """
    #from IPython import embed; embed()

    np_manual_node_feats, manual_node_keys = getmanuals(taus, tau1prods, tau2prods, events.MET, norm)
    tau_feats = ["pt", "ptCorrPNet", "eta", "phi", "mass", "dxy", "dz", 
                 "rawDeepTau2018v2p5VSe", "rawDeepTau2018v2p5VSjet", "rawDeepTau2018v2p5VSmu", 
                 "rawIsodR03", "rawMVAnewDM2017v2", "leadTkPtOverTauPt",
                 "rawPNetVSe", "rawPNetVSjet", "rawPNetVSmu", 
                 "charge@cat", "decayMode@cat", "decayModePNet@cat",
                 "probDM0PNet", "probDM10PNet", "probDM11PNet", "probDM1PNet", "probDM2PNet"]
    np_tau_node_feats, tau_node_keys = getnodes(taus, tau_feats, 2, "tau", norm) 
    jet_feats = ["pt", "eta", "phi", "mass", "btagPNetB", "btagDeepFlavB"]
    np_jet_node_feats, jet_node_keys = getnodes(jets, jet_feats, 15, "jet", norm)

    np_global_feats, global_keys = getglobals(global_var_to_keep(events, taus), norm)

    np_target_feats, target_keys = gettargets(gentaus, gentaunus)
    #np_target_feats = np.concatenate((np_target_feats, phi_cp), axis=1)
    #target_keys = target_keys + ["phicp"]
    #from IPython import embed; embed()
    extra_feats = ak.concatenate([phi_cp, 
                                  h1x, h1y, h1z, h2x, h2y, h2z,
                                  k1x, k1y, k1z, k2x, k2y, k2z, k1rx, k1ry, k1rz, k2rx, k2ry, k2rz], axis=1)
    extra_feats = ak.to_numpy(extra_feats)
    extra_keys = ["phicp", "h1x", "h1y", "h1z", "h2x", "h2y", "h2z", "k1x", "k1y", "k1z", "k2x", "k2y", "k2z", "k1rx", "k1ry", "k1rz", "k2rx", "k2ry", "k2rz"]

    #from IPython import embed; embed()

    all_feats = np.concatenate((np_tau_node_feats, 
                                np_jet_node_feats, 
                                np_manual_node_feats, 
                                np_global_feats, 
                                np_target_feats,
                                extra_feats), axis=1)
    all_keys  = tau_node_keys + jet_node_keys + manual_node_keys + global_keys + target_keys + extra_keys

    print(f" ---> feat shape: \n\t{np_tau_node_feats.shape}\n\t{np_jet_node_feats.shape}\n\t{np_manual_node_feats.shape}\n\t{np_global_feats.shape}\n\t{np_target_feats.shape}\n\t{extra_feats.shape}\n\t{all_feats.shape}")
    print(f" ---> nfeats    : \n\t{len(tau_node_keys)}\n\t{len(jet_node_keys)}\n\t{len(manual_node_keys)}\n\t{len(global_keys)}\n\t{len(target_keys)}\n\t{len(extra_keys)}\n\t{len(all_keys)}")
    #data  = pd.DataFrame(all_feats, columns=all_keys) 
    #print(list(data.keys()))
    #print(data.head(10))

    return all_feats, all_keys

    #return data

#def plotcols(_df):
#    variables = list(_df.keys())
    



def main():
    datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Pre-Processing')
    parser.add_argument('-n',
                        '--normalise',
                        action='store_true',
                        required=False,
                        default=False,
                        help="To save the normalised features: Todo -> To be implemented later with pytorch")
    parser.add_argument('-t',
                        '--tag',
                        type=str,
                        required=False,
                        default="Temp",
                        help="Make a dir with a tag to save the parquets with node, target and graph level features")
    parser.add_argument('-DM',
                        '--DecayMode',
                        type=str,
                        required=False,
                        default="",
                        help="Make a dir with a tag to save the parquets with node, target and graph level features")

    args = parser.parse_args()

    outdir = f"{args.tag}_{datetime_tag}" if not args.normalise else f"{args.tag}_norm_{datetime_tag}"

    config = getconfig('config.yml')
    infiles = getlistofinputs(config.get('indir'))
    assert len(infiles) > 0, 'no input files'
    catstokeep = config.get('catstokeep')

    events_df = None
    events_dict = {}
    
    #main_df = None
    main_np = None
    for index, fname in enumerate(infiles):
        print(f"\nFile idx: {index} --- File name: {fname}")
        #if (index > 0): continue
        assert os.path.exists(fname), f"{fname} doesnt exist"
        #events = NanoEventsFactory.from_root(
        #    {fname: "Events"},
        #    #steps_per_file=2_000,
        #    #metadata={"dataset": "DoubleMuon"},
        #    #schemaclass=NanoAODSchema,
        #).events()

        infile = uproot.open(fname)
        tree = infile['Events']
        events = NanoEventsFactory.from_root(fname).events()
        print(f"nEvents: {ak.num(events.event, axis=0)}")
        print(f"Event Fields: {events.fields}")
        events_feats, events_keys = get_events_dict(events, args.normalise, args.DecayMode)
        #events_df = get_events_dict(events, args.normalise)
        if index == 0:
            #from IPython import embed; embed()
            main_np = events_feats
        else:
            #from IPython import embed; embed()
            #main_df = pd.concat([main_df, events_df])
            main_np = np.concatenate((main_np, events_feats), axis=0)

    #from IPython import embed; embed()
    print(main_np.shape)
    main_df = pd.DataFrame(main_np, columns=events_keys) 
    print(list(main_df.keys()))
    print(main_df.head())

    print(f"nEntries: {main_df.shape}")
    h5name = f'GluGluHToTauTauSMdata_norm_{args.DecayMode}.h5' if args.normalise else f'GluGluHToTauTauSMdata_{args.DecayMode}.h5'
    main_df.to_hdf(h5name, key='df', mode='w')
    #return outdir, main_df



if __name__=="__main__":
    #from PrepareZtoTauTauDataset import ZtoTauTauDataset
    #outdir, events_dict = main()
    main()
