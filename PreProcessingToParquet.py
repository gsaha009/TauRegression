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
from tqdm.notebook import tqdm
import itertools
from typing import Optional
#import torch
#from torch_geometric.loader import DataLoader


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
    return sel_muo_indices


def electron_selection(events: ak.Array, mu_idxs=None) -> ak.Array:
    eles = events.Electron
    sorted_indices = ak.argsort(eles.pt, axis=-1, ascending=False)
    ele_mask = (
        (eles.pt > 25) &
        (np.abs(eles.eta) < 2.5) &
        (abs(eles.dxy) < 0.045) &
        (abs(eles.dz) < 0.2) &
        #(eles.mvaIso_WP80 == 1)
        (eles.mvaFall17V2Iso_WP80 == 1)
    )
    if mu_idxs is not None:
        ele_mask = ele_mask & ak.all(eles.metric_table(events.Muon[mu_idxs]) > 0.2, axis=2)
    
    sel_ele_indices = sorted_indices[ele_mask[sorted_indices]]
    return sel_ele_indices


def tau_selection(events: ak.Array, ele_idxs=None, mu_idxs=None) -> ak.Array:
    taus = events.Tau
    sorted_indices = ak.argsort(taus.pt, axis=-1, ascending=False)
    tau_mask = (
        (taus.pt > 20)
        & (abs(taus.eta) < 2.3)
        & (abs(taus.dz) < 0.2)
        & (taus.idDecayModeNewDMs)
        & ((taus.decayMode == 0)
           | (taus.decayMode == 1)
           | (taus.decayMode == 2)
           | (taus.decayMode == 10)
           | (taus.decayMode == 11))
        & (taus.idDeepTau2017v2p1VSjet >= 4) #4
        & (taus.idDeepTau2017v2p1VSe >= 2)
        & (taus.idDeepTau2017v2p1VSmu >= 1)
    )
    if ele_idxs is not None:
        tau_mask = tau_mask & ak.all(taus.metric_table(events.Electron[ele_idxs]) > 0.2, axis=2)
    if mu_idxs is not None:
        tau_mask = tau_mask & ak.all(taus.metric_table(events.Muon[mu_idxs]) > 0.2, axis=2)

    sel_tau_indices = sorted_indices[tau_mask[sorted_indices]]
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
    return sel_jet_indices



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
        print(i)
        temp = arr[:,i:i+1]
        temp_mean = ak.mean(temp)
        std_temp = ak.std(temp)
        temp_scaled = temp - temp_mean if std_temp == 0.0 else (temp - temp_mean)/std_temp
        print(f"scaled: {temp_scaled}")
        scaled_arr = temp_scaled if i == 0 else ak.concatenate([scaled_arr, temp_scaled], axis=ax)
    return scaled_arr
    
def filldict(dict={}, colname="", array=None, iteridx=0):
    if iteridx == 0:
        dict[colname] = ak.to_numpy(array).reshape(-1)
    else:
        np.append(dict[colname], ak.to_numpy(array).reshape(-1))

# ----------------------------------------------------------------------------------------------------- #
#                                            Node features                                              #
# ----------------------------------------------------------------------------------------------------- #
def getnode_dict_in(taus: ak.Array, jets: ak.Array, met: ak.Array):
    nodeDict = {
        "Tau": taus, 
        "Jet": jets,
        "MET": met
    }
    return nodeDict

def gettarget_dict_in(gens: ak.Array, gentaus: ak.Array):
    targetDict = {
        "GenNu": gens,
        #"GenTau": gentaus,
    }
    return targetDict


def getnodefeature(events: ak.Array,
                   tag: str="",
                   objcolDict: dict={},
                   fnames: list=[], 
                   fillNone: Optional[bool]=False,
                   StdScalar: Optional[bool]=False) -> ak.Array:
    print("\nGetting node features ...")
    tempdict = {}
    #templist = []
    for fname_ in fnames:
        iscat = True if "@cat" in fname_ else False
        fname = fname_.replace("@cat", "") if iscat else fname_
        templist = []
        for obj, col in objcolDict.items():
            print(f"\tField name: {obj}")
            if fname in col.fields:
                print(f'\t\t{fname} is found in the {obj} fields')
                temp = col[fname]
                if iscat:
                    temp = temp + 10 
                #if StdScalar:
                #    if not iscat:
                #        temp = scalefeat(temp)
                #print(ak.num(temp, axis=1))
                #tempdict[f"{tag}_{obj.lower()}_{i}_{fname}"] = ak.to_numpy(temp).reshape(-1)
            else:
                print(f'\t\t{fname} is NOT found in the {obj} fields')
                temp = ak.zeros_like(col["pt"], dtype=np.float32)
                #print(ak.num(temp, axis=1))
                #templist.append(temp)
                #print(temp)

            templist.append(temp)
        temp_concat = ak.concatenate(templist, axis=-1)
        #tempdict[f"{tag}_{fname}"] = temp_concat
        if iscat:
            unique_vals = np.unique(ak.ravel(temp_concat).to_numpy())
            for val in unique_vals:
                if val == 0: continue
                #lfrom IPython import embed; embed()
                #tempdict[f"{tag}_{fname}_{int(val)}"] = ak.where(temp_concat == val, temp_concat-10, ak.zeros_like(temp_concat))
                tempdict[f"{tag}_{fname}_{int(val)}"] = ak.where(temp_concat == int(val),
                                                                 ak.ones_like(temp_concat),
                                                                 ak.zeros_like(temp_concat))

        else:
            if StdScalar:
                print(temp_concat)
                temp_concat = scalefeat_nest(temp_concat, 1)
            tempdict[f"{tag}_{fname}"] = temp_concat
    return tempdict



def getmanualfeature()








def getnodefeature(col: ak.Array,
                   tag: str="",
                   fnames: list=[], 
                   fillNone: Optional[bool]=False,
                   StdScalar: Optional[bool]=False,
                   **kwargs) -> ak.Array:
    print("\nGetting node features ...")
    tempdict = {}
    #templist = []
    for fname_ in fnames:
        iscat = True if "@cat" in fname_ else False
        fname = fname_.replace("@cat", "") if iscat else fname_
        templist = []
        if fname in col.fields:
            print(f'\t\t{fname} is found in the {obj} fields')
            temp = col[fname]
            #if iscat:
            #    temp = temp + 10 
            #if StdScalar:
            #    if not iscat:
            #        temp = scalefeat(temp)
            #print(ak.num(temp, axis=1))
            #tempdict[f"{tag}_{obj.lower()}_{i}_{fname}"] = ak.to_numpy(temp).reshape(-1)
        else:
            print(f'\t\t{fname} is NOT found in the {obj} fields')
            temp = ak.zeros_like(col["pt"], dtype=np.float32)
            #print(ak.num(temp, axis=1))
            #templist.append(temp)
            #print(temp)
            
            templist.append(temp)
    temp_concat = ak.concatenate(templist, axis=-1)
    #tempdict[f"{tag}_{fname}"] = temp_concat
    if iscat:
        unique_vals = np.unique(ak.ravel(temp_concat).to_numpy())
        for val in unique_vals:
            if val == 0: continue
            #lfrom IPython import embed; embed()
            #tempdict[f"{tag}_{fname}_{int(val)}"] = ak.where(temp_concat == val, temp_concat-10, ak.zeros_like(temp_concat))
            tempdict[f"{tag}_{fname}_{int(val)}"] = ak.where(temp_concat == int(val),
                                                             ak.ones_like(temp_concat),
                                                             ak.zeros_like(temp_concat))
    else:
        if StdScalar:
            print(temp_concat)
            temp_concat = scalefeat_nest(temp_concat, 1)
        tempdict[f"{tag}_{fname}"] = temp_concat
    return tempdict

            

# ----------------------------------------------------------------------------------------------------- #
#                                          Global features                                              #
# ----------------------------------------------------------------------------------------------------- #
def getglobal_dict_in(events: ak.Array, taus: ak.Array):
    globalDict = {
        "met_pt": {"coll": events.MET, "fname": "pt"},
        "met_phi": {"coll": events.MET, "fname": "phi"},
        "met_covXX": {"coll": events.MET, "fname": "covXX"},
        "met_covXY": {"coll": events.MET, "fname": "covXY"},
        "met_covYY": {"coll": events.MET, "fname": "covYY"},
        "met_sumEt": {"coll": events.MET, "fname": "sumEt"},
        "met_significance": {"coll": events.MET, "fname": "significance"},        
        "nJet": {"coll": events.Jet, "fname": "num"},
        "nPV" :  {"coll": events.PV, "fname": "npvsGood"},
        "HT"  : {"coll": events.Jet.pt, "fname": "sum"},
        "MT_tau1_met" : {"coll": [taus[:,:1], events.MET], "fname": lambda x, y: get_MT(x,y)},
        "MT_tau2_met" : {"coll": [taus[:,1:2], events.MET], "fname": lambda x, y: get_MT(x,y)},
        "MT_tau1_tau2": {"coll": [taus[:,:1], taus[:,1:2]], "fname": lambda x, y: get_MT(x,y)},
        "MT_total": {"coll": [taus[:,:1], taus[:,1:2], events.MET], "fname": lambda x, y, z: get_total_MT(x,y,z)},
    }
    return globalDict


def getglobalfeature(events: ak.Array,
                     tag="global",
                     objcolDict = {},
                     fillNone=False,
                     StdScalar=False)->ak.Array:
    print("\nGetting global features ...")
    tempdict = {}

    for obj, coldict in objcolDict.items():
        print(f"\tField name: {obj}")
        temp = None
        if coldict["fname"] == "num":
            print(f"""\t\t{coldict["fname"]}: take the num objs""")
            temp = ak.fill_none(ak.num(coldict["coll"], axis=1), 0.0)
            #if StdScalar:
            #    temp = scalefeat(temp)
            #print(temp)
            #tempdict[f"{obj.lower()}"] = ak.flatten(temp, axis=None)

        elif coldict["fname"] == "sum":
            print(f"""\t\t{coldict["fname"]}: take the sum of objs""")
            temp = ak.fill_none(ak.sum(coldict["coll"], axis=1), 0.0)
            #if StdScalar:
            #    temp = scalefeat(temp)
            #print(temp)
            #tempdict[f"{obj.lower()}"] = ak.flatten(temp, axis=None)

        elif isinstance(coldict["fname"], types.LambdaType):
            print(f"""\t\t{coldict["fname"]}: call lambda""")
            temp = ak.fill_none(coldict["fname"](*coldict["coll"]), 0.0)
            #if StdScalar:
            #    temp = scalefeat(temp)
            #print(temp)
            #tempdict[f"{obj.lower()}"] = ak.flatten(temp, axis=None)

        else:
            assert coldict["fname"] in coldict["coll"].fields, f"feat must be in the fields"
            print(f"""\t\t{coldict["fname"]}: found in the {obj} fields""")
            temp = ak.fill_none(coldict["coll"][coldict["fname"]], 0.0)
            #if StdScalar:
            #    temp = scalefeat(temp)
            #print(temp)
            #tempdict[f"{obj.lower()}"] = ak.flatten(temp, axis=None)
        print(f"global: {temp}")
        if StdScalar:
            temp = scalefeat_flat(temp)
        tempdict[f"{tag}_{obj.lower()}"] = ak.flatten(temp, axis=None)
    #print(f"global dict: {tempdict}")
    #return {f"{tag}_feats": tempdict}
    return tempdict




# ----------------------------------------------------------------------------------------------------- #
#                                            main script                                                #
# ----------------------------------------------------------------------------------------------------- #
def get_events_dict(events:ak.Array, scale: bool):

    muo_sel_idx = muon_selection(events)
    ele_sel_idx = electron_selection(events, mu_idxs=muo_sel_idx)
    tau_sel_idx = tau_selection(events, ele_idxs=ele_sel_idx, mu_idxs=muo_sel_idx)
    jet_sel_idx = jet_selection(events, ele_idxs=ele_sel_idx, mu_idxs=muo_sel_idx, tau_idxs=tau_sel_idx)
    gentaunu_sel_idx = gentaunu_selection(events, tau_idxs=tau_sel_idx)
    gentau_sel_idx = gentau_selection(events, tau_idxs=tau_sel_idx)

        
    is_tauhtauh = (
        (ak.num(tau_sel_idx, axis=1) == 2)
        & (ak.num(ele_sel_idx, axis=1) == 0)
        & (ak.num(muo_sel_idx, axis=1) == 0)
    )

    is_os = ak.sum(events.Tau[tau_sel_idx].charge, axis=-1) == 0
    #print(f"is_os: {is_os}")
    has_two_gentaunu = ak.num(gentaunu_sel_idx, axis=-1) >= 2
    has_two_gentau   = ak.num(gentau_sel_idx, axis=-1) >= 2
    invmass = invariant_mass(events.Tau[tau_sel_idx])
    #print(f"invmass: {invmass}")
    isZ = (invmass > 12) & (invmass < 200)
    #print(f"isZ: {isZ}")
    #from IPython import embed; embed()
    
    false_mask = (abs(events.event) < 0)
    #channel_id = np.uint8(1) * false_mask
    empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    
    where = is_tauhtauh & is_os & has_two_gentaunu & has_two_gentau & isZ

    print(f"is_tauhtauh: {ak.sum(is_tauhtauh)}")
    print(f"is_tauhtauh & is_os: {ak.sum(is_tauhtauh & is_os)}")
    print(f"is_tauhtauh & is_os & has_two_gentau: {ak.sum(is_tauhtauh & is_os & has_two_gentau)}")
    print(f"is_tauhtauh & is_os & has_two_gentau & isZ: {ak.sum(is_tauhtauh & is_os & has_two_gentau & isZ)}")

    
    muo_sel_idx = ak.where(where, muo_sel_idx, empty_indices)
    ele_sel_idx = ak.where(where, ele_sel_idx, empty_indices)
    tau_sel_idx = ak.where(where, tau_sel_idx, empty_indices)
    jet_sel_idx = ak.where(where, jet_sel_idx, empty_indices)
    gentaunu_sel_idx = ak.where(where, gentaunu_sel_idx, empty_indices)
    gentau_sel_idx = ak.where(where, gentau_sel_idx, empty_indices)
    
    muons = events.Muon[muo_sel_idx]
    electrons = events.Electron[ele_sel_idx]
    jets = events.Jet[jet_sel_idx]
    taus = events.Tau[tau_sel_idx]
    gentaunus = events.GenPart[gentaunu_sel_idx]
    gentaus = events.GenPart[gentau_sel_idx]

    print(f"Fields: {gentaunus.fields}, {gentaus.fields}")
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
    
    # event and object masks applied
    events = events[where]

    # indices
    muo_sel_idx = muo_sel_idx[where]
    ele_sel_idx = ele_sel_idx[where]
    tau_sel_idx = tau_sel_idx[where]
    jet_sel_idx = jet_sel_idx[where]
    gentaunu_sel_idx = gentaunu_sel_idx[where]
    gentau_sel_idx = gentau_sel_idx[where]
    # objects
    muons = muons[where]
    electrons = electrons[where]
    jets = jets[where]
    taus = taus[where]
    gentaunus = taus.nearest(gentaunus[where], threshold=0.4)
    gentaus = taus.nearest(gentaus[where], threshold=0.4)
    tauprods = tauprods[where]
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
    
    in_nodeDict = getnode_dict_in(taus, jets)
    in_edgeDict = getedge_dict_in(taus, jets)
    in_targetDict = gettarget_dict_in(gentaunus, gentaus)
    in_globalDict = getglobal_dict_in(events, taus)

    events_dict = {}
    
    #edge_dict = getedgefeature(events, tag="edge", colDict = in_edgeDict)
    
    node_types = ["pt", "eta", "phi", "mass", "dxy", "dz", "rawDeepTau2017v2p1VSe", "rawDeepTau2017v2p1VSjet",
                  "rawDeepTau2017v2p1VSmu", "rawIsodR03", "rawMVAnewDM2017v2", "leadTkPtOverTauPt", "btagDeepFlavB",
                  "charge@cat", "decayMode@cat"]
    
    node_dict = getnodefeature(events, tag="node", objcolDict=in_nodeDict, fnames=node_types, StdScalar=scale)
    print(f"node_dict: {node_dict}")
    
    target_types = ["px", "py", "pz"]
    target_dict = getnodefeature(events, tag="target", objcolDict=in_targetDict, fnames=target_types)
    print(f"target_dict: {target_dict}")
    
    
    global_dict = getglobalfeature(events, tag="global", objcolDict=in_globalDict, StdScalar=scale)
    print(f"global dict: {global_dict}")
    
    #edge_idxs, edgefeature_dr = getedgefeature(events, colDict=edgeDict)
    
    events_dict.update(node_dict)
    events_dict.update(global_dict)
    ##events_dict.update(edge_dict)
    events_dict.update(target_dict)
    
    #events_df = pd.DataFrame.from_dict(events_dict) if index == 0 else pd.concat([events_df, pd.DataFrame.from_dict(events_dict)])
    
    #if index == 0:
    #events_df = pd.concat([events_df, pd.DataFrame.from_dict(events_dict)])
    
    #print(events_dict)
    print(f"nEvents final Selected: {ak.num(events.event, axis=0)}")
    
    return events_dict


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
    args = parser.parse_args()

    outdir = f"{args.tag}_{datetime_tag}" if not args.normalise else f"{args.tag}_norm_{datetime_tag}"

    config = getconfig('gnnconfig.yml')
    infiles = getlistofinputs(config.get('indir'))
    assert len(infiles) > 0, 'no input files'
    catstokeep = config.get('catstokeep')

    events_df = None
    events_dict = {}
    
    for index, fname in enumerate(infiles):
        print(f"\nFile idx: {index} --- File name: {fname}")
        #if (index > 1): continue
        assert os.path.exists(fname), f"{fname} doesnt exist"
        infile = uproot.open(fname)
        ##print(infile.keys())
        tree = infile['Events']
        events = NanoEventsFactory.from_root(fname).events()
        print(f"nEvents: {ak.num(events.event, axis=0)}")

        temp_dict = get_events_dict(events, args.normalise)
        print(temp_dict.keys())
        if (index == 0):
            events_dict = temp_dict
        else:
            #from IPython import embed; embed()            
            #events_dict["node_feats"] = {key:ak.concatenate([events_dict["node_feats"][key], val]) for key,val in temp_dict["node_feats"].items()}
            #events_dict["target_feats"] = {key:ak.concatenate([events_dict["target_feats"][key], val]) for key,val in temp_dict["target_feats"].items()}
            #events_dict["global_feats"] = {key:ak.concatenate([events_dict["global_feats"][key], val]) for key,val in temp_dict["global_feats"].items()}
            events_dict.update({key:ak.concatenate([events_dict[key], val]) for key,val in temp_dict.items() if key.split('_')[0] == "node"})
            events_dict.update({key:ak.concatenate([events_dict[key], val]) for key,val in temp_dict.items() if key.split('_')[0] == "target"})
            events_dict.update({key:ak.concatenate([events_dict[key], val]) for key,val in temp_dict.items() if key.split('_')[0] == "global"})

        #events_df = pd.DataFrame.from_dict(events_dict) if index == 0 else pd.concat([events_df, pd.DataFrame.from_dict(events_dict)], ignore_index=True)

    #print(f"events_dict: {events_dict}")
    #global_dict = events_dict["global_feats"]
    #for key, val in global_dict.items():
    #    print(f"{key}: \t{ak.count(val)}")
    #target_dict = events_dict["target_feats"]
    #for key, val in target_dict.items():
    #    print(f"{key}: \t{ak.num(val, axis=1)} \t{ak.num(val, axis=0)}")
    #node_dict = events_dict["node_feats"]
    #for key, val in node_dict.items():
    #    print(f"{key}: \t{ak.num(val, axis=1)} \t{ak.num(val, axis=0)}")
    return outdir, events_dict

if __name__=="__main__":
    #from PrepareZtoTauTauDataset import ZtoTauTauDataset
    outdir, events_dict = main()
    #dataset = ZtoTauTauDataset(events_dict)
    #print(dataset[0])
    #print(dataset[0], dataset.len())
    #for i in range(dataset.len()):
    #    print(dataset[i])

    node_arrays = ak.zip({key.replace("node_",""):val for key,val in events_dict.items() if key.split("_")[0] == "node"})
    print(node_arrays.fields)
    #for key in node_arrays.fields: print(node_arrays[key])
    target_arrays = ak.zip({key.replace("target_",""):val for key,val in events_dict.items() if key.split("_")[0] == "target"})
    print(target_arrays.fields)
    #for key in target_arrays.fields: print(target_arrays[key])
    global_arrays = ak.zip({key.replace("global_",""):val for key,val in events_dict.items() if key.split("_")[0] == "global"})
    print(global_arrays.fields)
    #for key in global_arrays.fields: print(global_arrays[key])
    
    #events_record = ak.Record(events_dict)

    ##print(events_record)
    ##print(events_record.fields)
    ##print(events_record["node_feats"])
    ##print(events_record["target_feats"])
    ##print(events_record["global_feats"])

    os.makedirs(outdir, exist_ok=True)
    
    ak.to_parquet(node_arrays, f"{outdir}/DY_nodes.parquet")
    ak.to_parquet(target_arrays, f"{outdir}/DY_targets.parquet")
    ak.to_parquet(global_arrays, f"{outdir}/DY_globals.parquet")
    
    #torch.save(dataset, 'dataset.pt')
    #dataset.save("dataset.pt")

    #dfname = "DY_df_temp.h5"
    ##df = pd.DataFrame.from_dict(events_dict)
    #events_df.to_hdf(dfname, key='df')
