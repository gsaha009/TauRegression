import os
import pandas as pd
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from IPython import embed
import awkward as ak
import vector as npvector
from coffea.nanoevents.methods import vector as akvector
from PhiCPComp import PhiCPComp
import datetime

datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#OD = "/sps/cms/gsaha/TauRegression/GNN_Output/Output_20240412_104117/Modified_GluGluHToTauTauSMdata_ForTrain_Raw_11_simple_IC_20240411_124244.h5"
#PD = "/sps/cms/gsaha/TauRegression/GNN_Output/Output_20240412_111918/ypreds_20240412_111918.h5"
OD = "/sps/cms/gsaha/TauRegression/GNN_Output/Output_20240417_062630/Modified_GluGluHToTauTauSMdata_ForTrain_Raw_11_simple_IC_20240416_133720.h5"
PD = "/sps/cms/gsaha/TauRegression/GNN_Output/Output_20240417_062630/ypreds_20240417_110203.h5"

outdir = os.path.dirname(PD)

df_OD = pd.read_hdf(OD)
df_PD = pd.read_hdf(PD)

print(df_OD.head())
print(df_PD.head())


def getakp4(p4, pid):
    return ak.zip(
        {
            "pt": ak.Array(p4.pt[:,None]),
            "eta": ak.Array(p4.eta[:,None]),
            "phi": ak.Array(p4.phi[:,None]),
            "mass": ak.Array(p4.mass[:,None]),
            "pdgId": pid*ak.ones_like(ak.Array(p4.mass[:,None])),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=akvector.behavior
    )


def create_p4(df, col_names, type="ptetaphim"):
    if type == "ptetaphim" :
        assert len(col_names) == 4
    else:
        assert len(col_names) == 3
    X  = df[col_names[0]].to_numpy()
    Y  = df[col_names[1]].to_numpy()
    Z  = df[col_names[2]].to_numpy()
    T  = df[col_names[3]].to_numpy() if type == "ptetaphim" else np.sqrt(X*X + Y*Y + Z*Z)

    if type == "ptetaphim":
        return npvector.array({"pt":  X, 
                               "eta": Y,
                               "phi": Z,
                               "M":   T})
    else:
        return npvector.array({"px": X, 
                               "py": Y,
                               "pz": Z,
                               "E":  T})
        


cols_OD = ["phicp", "phi_cp_det",
           "pt_tau_1", "eta_tau_1", "phi_tau_1", "mass_tau_1",
           "pt_tau_2", "eta_tau_2", "phi_tau_2", "mass_tau_2",
           "pt_tau1pi_1", "eta_tau1pi_1", "phi_tau1pi_1", "mass_tau1pi_1",
           "pt_tau1pi0_1", "eta_tau1pi0_1", "phi_tau1pi0_1", "mass_tau1pi0_1",
           "pt_tau2pi_1", "eta_tau2pi_1", "phi_tau2pi_1", "mass_tau2pi_1",
           "pt_tau2pi0_1", "eta_tau2pi0_1", "phi_tau2pi0_1", "mass_tau2pi0_1",
]
cols_PD = ["px_gentaunu_1", "py_gentaunu_1", "pz_gentaunu_1",
           "px_gentaunu_2", "py_gentaunu_2", "pz_gentaunu_2"]


df_OD = df_OD[cols_OD]
df_PD = df_PD[cols_PD]


print(df_OD.head())
print(df_PD.head())


phicp_full_gen = df_OD["phicp"].to_numpy()
phicp_gen_det  = df_OD["phi_cp_det"].to_numpy()

tau1p4 = create_p4(df_OD, ["pt_tau_1", "eta_tau_1", "phi_tau_1", "mass_tau_1"])
tau2p4 = create_p4(df_OD, ["pt_tau_2", "eta_tau_2", "phi_tau_2", "mass_tau_2"])
tau1_pi_p4 = create_p4(df_OD, ["pt_tau1pi_1", "eta_tau1pi_1", "phi_tau1pi_1", "mass_tau1pi_1"])
tau1_pi0_p4 = create_p4(df_OD, ["pt_tau1pi0_1", "eta_tau1pi0_1", "phi_tau1pi0_1", "mass_tau1pi0_1"])
tau2_pi_p4 = create_p4(df_OD, ["pt_tau2pi_1", "eta_tau2pi_1", "phi_tau2pi_1", "mass_tau2pi_1"])
tau2_pi0_p4 = create_p4(df_OD, ["pt_tau2pi0_1", "eta_tau2pi0_1", "phi_tau2pi0_1", "mass_tau2pi0_1"])

regr_nu1_p4 = create_p4(df_PD, ["px_gentaunu_1", "py_gentaunu_1", "pz_gentaunu_1"], type="pxpypze")
regr_nu2_p4 = create_p4(df_PD, ["px_gentaunu_2", "py_gentaunu_2", "pz_gentaunu_2"], type="pxpypze")

tau1p4_full = tau1p4 + regr_nu1_p4
tau2p4_full = tau2p4 + regr_nu2_p4

tau1p4_full = getakp4(tau1p4_full, 15)
tau2p4_full = getakp4(tau2p4_full, 15)
tau1_pi_p4  = getakp4(tau1_pi_p4, 211)
tau2_pi_p4  = getakp4(tau2_pi_p4, 211)
tau1_pi0_p4 = getakp4(tau1_pi0_p4, 111)
tau2_pi0_p4 = getakp4(tau2_pi0_p4, 111)


print(f"askmclkmlkmlkmclksa: {tau1_pi_p4.pt}")
print(f"askmclkmlkmlkmclksa: {tau1_pi0_p4.pt}")
print(f"askmclkmlkmlkmclksa: {tau2_pi_p4.pt}")
print(f"askmclkmlkmlkmclksa: {tau2_pi0_p4.pt}")

tau1_decay = ak.from_regular(ak.concatenate([tau1_pi_p4, tau1_pi0_p4], axis=1))
tau2_decay = ak.from_regular(ak.concatenate([tau2_pi_p4, tau2_pi0_p4], axis=1))

#embed()


phicp_obj = PhiCPComp(cat="rhorho",
                      taum=tau1p4_full,
                      taup=tau2p4_full,
                      taum_decay=tau1_decay,
                      taup_decay=tau2_decay)
phi_cp, _ = phicp_obj.comp_phiCP()
#phi_cp = ak.firsts(phi_cp, axis=1)
print(phi_cp)

#phicp_full_gen = df_OD["phicp"].to_numpy()
#phicp_gen_det  = df_OD["phi_cp_det"].to_numpy()
np_phi_cp_det = ak.to_numpy(phi_cp).reshape(-1)
print(phicp_full_gen, phicp_gen_det, np_phi_cp_det)

#phicpdf = ak.to_dataframe(phi_cp)
phicpdf = pd.DataFrame(np_phi_cp_det, columns=["PhiCP_Det"])
print(phicpdf.head())
phicpdf.to_hdf(os.path.join(outdir, f"PhiCP_{datetime_tag}.h5"), key="df", mode="w")

print("Plotting ...")

plt.figure(figsize=(10,8))

plt.hist(np_phi_cp_det, bins=25, label="Full Det", density=True, alpha=0.9, lw=2, histtype="step")
plt.hist(phicp_gen_det, bins=25, label="Gen tau + det decays", density=True, alpha=0.9, lw=2, histtype="step")
plt.hist(phicp_full_gen, bins=25, label="Full Gen", density=True, alpha=0.9, lw=2, histtype="step")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(outdir, f"PhiCP_{datetime_tag}.png"), dpi=300)

print("DONE")
