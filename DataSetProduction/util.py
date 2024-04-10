import os
import sys
import yaml
import glob
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from typing import Optional
from coffea.nanoevents.methods import vector


class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

def setp4_(arr: ak.Array, mass: float):
    return ak.zip(
        {
            "pt": arr.pt,
            "eta": arr.eta,
            "phi": arr.phi,
            "mass": mass*ak.ones_like(arr.pt),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ) 

                
def setp4(name="LorentzVector", *args, verbose: Optional[bool] = False):
    if len(args) < 4:
        raise RuntimeError ("Need at least four components")
    
    if name == "PtEtaPhiMLorentzVector":
        if verbose:
            print(f" --- pT  : {args[0]}")
            print(f" --- eta : {args[1]}")
            print(f" --- phi : {args[2]}")
            print(f" --- mass: {args[3]}")         
        return ak.zip(
            {
                "pt": args[0],
                "eta": args[1],
                "phi": args[2],
                "mass": args[3],
                "pdgId": args[4] if len(args) > 4 else -99
            },
            with_name=name,
            behavior=vector.behavior
        )
    
    else:
        if verbose:
            print(f" --- px     : {args[0]}")
            print(f" --- py     : {args[1]}")
            print(f" --- pz     : {args[2]}")
            print(f" --- energy : {args[3]}")
        return ak.zip(
            {
                "x": args[0],
                "y": args[1],
                "z": args[2],
                "t": args[3],
                "pdgId": args[4] if len(args) > 4	else -99
            },
            with_name=name,
            behavior=vector.behavior
        )
    
    
    
def getp4(genarr: ak.Array, verbose: Optional[bool] = False) -> ak.Array:
    if verbose:
        print(f" --- pT  : {genarr.pt}")
        print(f" --- eta : {genarr.eta}")
        print(f" --- phi : {genarr.phi}")
        print(f" --- mass: {genarr.mass}")
    return ak.zip(
        {
            "pt": genarr.pt,
            "eta": genarr.eta,
            "phi": genarr.phi,
            "mass": genarr.mass,
            "pdgId": genarr.pdgId if "pdgId" in genarr.fields else -99.9*ak.ones_like(genarr.pt),
            "tauIdx": genarr.pdgId if "tauIdx" in genarr.fields else -99.9*ak.ones_like(genarr.pt),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

        
def plotit(arrlist=[], bins=100, log=False, dim=(1,1)):
    print(" ---> Plotting ---> ")
    dim = (1, len(arrlist))
    if len(arrlist) > 1 :
        fig, axs = plt.subplots(dim[0], dim[1], figsize=(2.5*dim[1], 2.5))
        for i,arr in enumerate(arrlist):
            axs[i].hist(arr, bins, density=False, log=log, alpha=0.7)
    else:
        fig, ax = plt.subplots()
        ax.hist(arrlist[0], bins, density=False, log=log, alpha=0.7)

    fig.tight_layout()
    plt.show()


def printinfo(p4, type="p", plot=True, log=False):
    if type=="p":
        print(f"    px: {p4.px}")
        print(f"    py: {p4.py}")
        print(f"    pz: {p4.pz}")
        print(f"    E: {p4.energy}")
        if plot:
            #plotit(arrlist=[ak.ravel(p4.px).to_numpy(),
            #                ak.ravel(p4.py).to_numpy(),
            #                ak.ravel(p4.pz).to_numpy(),
            #                ak.ravel(p4.energy).to_numpy()],
            #       bins=50,
            #       log=log)
            plotit(arrlist=[ak.fill_none(ak.flatten(p4.px),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.py),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.pz),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.energy),0).to_numpy()],
                   bins=50,
                   log=log)

    elif type=="pt":
        print(f"    pt: {p4.pt}")
        print(f"    eta: {p4.eta}")
        print(f"    phi: {p4.phi}")
        print(f"    M: {p4.mass}")
        if plot:
            #plotit(arrlist=[ak.ravel(p4.pt).to_numpy(),
            #                ak.ravel(p4.eta).to_numpy(),
            #                ak.ravel(p4.phi).to_numpy(),
            #                ak.ravel(p4.mass).to_numpy()],
            #       bins=50,
            #       log=log)
            plotit(arrlist=[ak.fill_none(ak.flatten(p4.pt),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.eta),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.phi),0).to_numpy(),
                            ak.fill_none(ak.flatten(p4.mass),0).to_numpy()],
                   bins=50,
                   log=log)

    else:
        print(f"    x: {p4.x}")
        print(f"    y: {p4.y}")
        print(f"    z: {p4.z}")
        print(f"    t: {p4.t}")
        if plot:
            plotit(arrlist=[ak.ravel(p4.x).to_numpy(),
                            ak.ravel(p4.y).to_numpy(),
                            ak.ravel(p4.z).to_numpy(),
                            ak.ravel(p4.t).to_numpy()],
                   bins=50,
                   log=log)
            

def printinfoP3(p3, plot=True, log=False):
    print(f"    x: {p3.x}")
    print(f"    y: {p3.y}")
    print(f"    z: {p3.z}")
    if plot:
        plotit(arrlist=[ak.ravel(p3.x).to_numpy(),
                        ak.ravel(p3.y).to_numpy(),
                        ak.ravel(p3.z).to_numpy()],
               bins=50,
               log=log)
        
    
def inspect(arr):
    print("....ispecting....")
    print(f"  dimension: {arr.ndim}")
    print(f"  type: {arr.type}")
    ne = np.count(arr) if arr.ndim == 1 else np.sum(arr, axis=0)
    print(f"  nEntries: {ne}")
    

def RotateX(tovect, theta):
    print("  --- RotateX ---  ")
    x = tovect.x
    y = tovect.y*np.cos(theta) - tovect.z*np.sin(theta)
    z = tovect.y*np.sin(theta) + tovect.z*np.cos(theta)
    t = tovect.t
    print(f"\t\t x: {x}")
    print(f"\t\t y: {y}")
    print(f"\t\t z: {z}")
    print(f"\t\t t: {t}")
    vec = ak.zip(
        {
            "x": x,
            "y": y,
            "z": z,
            "t": t,
        },
            with_name="LorentzVector",
        behavior=vector.behavior,
    )
    return vec


def RotateY(tovect, theta):
    print("  --- RotateY ---  ")
    x = tovect.x*np.cos(theta) + tovect.z*np.sin(theta)
    y = tovect.y
    z = tovect.z*np.cos(theta) - tovect.x*np.sin(theta)
    t = tovect.t
    print(f"\t\t x: {x}")
    print(f"\t\t y: {y}")
    print(f"\t\t z: {z}")
    print(f"\t\t t: {t}")
    vec = ak.zip(
        {
            "x": x,
            "y": y,
            "z": z,
            "t": t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    return vec

def RotateZ(tovect, theta):
    print("  --- RotateZ ---  ")
    x = tovect.x*np.cos(theta) - tovect.y*np.sin(theta)
    y = tovect.x*np.sin(theta) + tovect.y*np.cos(theta)
    z = tovect.z
    t = tovect.t
    print(f"\t\t x: {x}")
    print(f"\t\t y: {y}")
    print(f"\t\t z: {z}")
    print(f"\t\t t: {t}")
    vec = ak.zip(
        {
            "x": x,
            "y": y,
            "z": z,
            "t": t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    return vec

def Rotate(vect, rotvect):
    print(" --- Rotate --- : --- Start ---")
    print(f"  ||| Before rotation - Mag2: {vect.mass2}")
    print(f"\t\t x: {vect.x}")
    print(f"\t\t y: {vect.y}")
    print(f"\t\t z: {vect.z}")
    print(f"\t\t t: {vect.t}")
    
    print(f"  rotvect.phi : {rotvect.phi}")
    print(f"  rotvect.theta : {rotvect.theta}")
    
    vect = RotateZ(vect, (0.5*np.pi - rotvect.phi))
    vect = RotateX(vect, rotvect.theta)
    print(f"  ||| After rotation - Mag2: {vect.mass2}")
    
    print(" --- Rotate --- : --- End ---\n")
    return vect

    
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


def plotdf(df, outf):
    keys = list(df.keys())
    nkeys = len(keys)
    ncols = 10
    nrows = int(np.ceil(nkeys/10))

    plt.figure(figsize=(4*ncols,2*nrows))
    for idx, key in enumerate(keys):
        idx=idx+1
        ax = plt.subplot(nrows,ncols,idx)
        arr = df[key]
        
        ax.hist(arr, 50, histtype="stepfilled", alpha=0.7, log=False)
        
        ax.set_title(f"{key}")
        ax.set_xlabel(f"{key}")
        #ax.legend()

    plt.tight_layout()
    plt.savefig(f'{outf}', dpi=300)
