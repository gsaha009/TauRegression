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

from util import *
from PhiCPComp import PhiCPComp
from EventSelection import SelectEvents
from ExtractFeatures import FeatureExtraction


# ----------------------------------------------------------------------------------------------------- #
#                                            main script                                                #
# ----------------------------------------------------------------------------------------------------- #
def get_features_df(events, tau1dm, tau2dm, isForTrain, howtogetpizero, isnorm):

    sel_evt_object = SelectEvents(events = events,
                                  tau1dm = tau1dm,
                                  tau2dm = tau2dm,
                                  isForTrain = isForTrain,
                                  howtogetpizero = howtogetpizero)

    sel_evt_dict = sel_evt_object.apply_selections()

    feat_extract_object = FeatureExtraction(sel_evt_dict["events"],
                                            sel_evt_dict["det_taus"],
                                            sel_evt_dict["det_tauprods_concat"],
                                            sel_evt_dict["gen_taus"],
                                            sel_evt_dict["gen_tauprods"],
                                            sel_evt_dict["gen_taunus"],
                                            sel_evt_dict["jets"],
                                            isnorm = isnorm,
                                            isForTrain = isForTrain)
    
    np_feats, np_keys       = feat_extract_object.extraction()
    extra_feats, extra_keys = feat_extract_object.extrafeats(tau1dm, tau2dm, sel_evt_dict["det_pions"], sel_evt_dict["det_pizeros"])
    
    all_feats = np.concatenate((np_feats, extra_feats), axis=1)
    all_keys  = np_keys + extra_keys

    df = pd.DataFrame(all_feats, columns=all_keys)

    return df



def main():
    datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Pre-Processing')
    parser.add_argument('-t',
                        '--tag',
                        type=str,
                        required=False,
                        default="ProdDataSet",
                        help="Make a dir with a tag to save the parquets with node, target and graph level features")

    args = parser.parse_args()

    config = getconfig('config.yml')
    infiles = getlistofinputs(config.get('indir'))
    assert len(infiles) > 0, 'no input files'
    outdir = config.get("outdir")
    isnorm = config.get("isnorm")
    tau1dm = config.get("tau1dm")
    tau2dm = config.get("tau2dm")
    isForTrain = config.get("isForTrain")
    howtogetpizero = config.get("howtogetpizero")

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        print(f"{outdir} exists")
    

    events_df = None
    events_dict = {}
    
    for index, fname in enumerate(infiles):
        print(f"\nFile idx: {index} --- File name: {fname}")
        #if (index > 0): continue
        assert os.path.exists(fname), f"{fname} doesnt exist"

        infile = uproot.open(fname)
        tree = infile['Events']
        events = NanoEventsFactory.from_root(fname).events()
        print(f"nEvents: {ak.num(events.event, axis=0)}")
        print(f"Event Fields: {events.fields}")

        df = get_features_df(events, tau1dm, tau2dm, isForTrain, howtogetpizero, isnorm)

        if index == 0:
            #from IPython import embed; embed()
            events_df = df
        else:
            events_df = pd.concat([events_df, df])

    #from IPython import embed; embed()
    print(list(events_df.keys()))
    print(events_df.head(10))

    print(f"nEntries: {events_df.shape}")
    _item_1 = "ForTrain" if isForTrain else "ForEval"
    _item_2 = "Norm" if isnorm else "Raw"
    _item_3 = f"{tau1dm}{tau2dm}"
    
    _name = f'GluGluHToTauTauSMdata_{_item_1}_{_item_2}_{_item_3}_{howtogetpizero}_{datetime_tag}'

    pltname = os.path.join(outdir, f"{_name}.pdf")
    h5name  = os.path.join(outdir, f"{_name}.h5")

    print("plotting df features ===> ")
    plotdf(events_df, pltname)
    print("saving dataframe ===>")
    events_df.to_hdf(h5name, key='df', mode='w')
    print("DONE")

if __name__=="__main__":
    main()
