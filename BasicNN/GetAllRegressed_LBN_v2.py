#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import datetime
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
from tqdm.notebook import tqdm
import vector
import itertools
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict
import seaborn as sns

import tensorflow as tf
import tensorflow.saved_model
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Add, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, model_from_json, load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from lbn import LBN, LBNLayer


# In[2]:


print(tf.__version__)
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)


train = True
evaluate = True
#df_ = "GluGluHToTauTauSMdata_ForTrain_Raw_11_simple_MB_20240411_115734.h5"
df_ = "GluGluHToTauTauSMdata_ForTrain_Raw_11_simple_IC_20240416_133720.h5"
#df_ = "GluGluHToTauTauSMdata_11_det.h5"


# In[4]:


outpath = "/sps/cms/gsaha/TauRegression/LBN_Output"

tag_ = "LBNResNet_Higgs_custom_2024_v1"
tagdir = os.path.join(os.getcwd(), tag_)
if not os.path.exists(tagdir):
    os.mkdir(tagdir)
else:
    print(f"{tag_} exists")


# In[5]:


# Load the TensorBoard notebook extension
#%reload_ext tensorboard
#from tensorboard import notebook
#notebook.list() # View open TensorBoard instances
## Control TensorBoard display. If no port is provided, 
## the most recently launched TensorBoard is used
#notebook.display(port=6006, height=1000)


# In[6]:


def PlotHistory(history, path=None):
    """ Takes history from Keras training and makes loss plots (batch and epoch) and learning rate plots """
    #----- Figure -----#
    variables = sorted([key for key in history.epochs.keys() if 'val' not in key and 'val_'+key in history.epochs.keys()])
    variables += ["lr"]
    N = len(variables)
    print(f"variables: {variables}")
    fig, ax = plt.subplots(N,2,figsize=(12.5,N*2),sharex='col')
    plt.subplots_adjust(left    = 0.1,
                        right   = 0.6,
                        top     = 0.9,
                        bottom  = 0.1,
                        hspace  = 0.5,
                        wspace  = 0.4)
    
    #----- Batch Plots -----#
    for i,var in enumerate(variables):
        ax[i,0].plot(history.batches['batch'],history.batches[var],'k')
        ax[i,0].set_title(var)
        ax[i,0].set_xlabel('Batch')
        
    #----- Epoch Plots -----#
    for i,var in enumerate(variables):
        ax[i,1].plot(history.epochs['epoch'],history.epochs[var],label='train')
        if 'val_'+var in history.epochs.keys():
            ax[i,1].plot(history.epochs['epoch'],history.epochs['val_'+var],label='validation')
        ax[i,1].set_title(var)
        ax[i,1].legend()
        ax[i,1].set_xlabel('Epoch')

    png_name = 'Loss_Acc_LR.png'
    fig.savefig(os.path.join(path, png_name), dpi=300)
    print('Curves saved as %s'%png_name)
#######################################################


def getp4(df, vars=[], LVtype=""):
    """
        df: The main dfataframe
        vars = [p4 components of obj]
        LVtype = ptetaphim / pxpypze
    """
    for var in vars:
        if var not in list(df.keys()):
            df[var] = 0.0
    p4 = None
    if LVtype == "ptetaphim":
        p4 = vector.array({"pt": df[vars[0]].to_numpy(),
                           "eta": df[vars[1]].to_numpy(),
                           "phi": df[vars[2]].to_numpy(),
                           "M": df[vars[3]].to_numpy()})
    elif LVtype == "pxpypze":
        p4 = vector.array({"px": df[vars[0]].to_numpy(),
                           "py": df[vars[1]].to_numpy(),
                           "pz": df[vars[2]].to_numpy(),
                           "E": df[vars[3]].to_numpy()})
        

    return p4

def setp4(df, vars=[]):
    p4 = getp4(df, vars=vars)
    obj = vars[0].split('_')[0]
    num = vars[0].split('_')[1]
    key = f'{obj}_{num}' if len(vars[0].split('_')) > 2 else f'{obj}'

    df[f'{key}_px'] = p4.px
    df[f'{key}_py'] = p4.py
    df[f'{key}_pz'] = p4.pz
    df[f'{key}_E']  = p4.E


# In[7]:


#indf = "df_DY_v5_processed_scaled_metphi.h5"
#indf = "df_DY_v5_processed.h5"
indf = f"/sps/cms/gsaha/GluGluHToTauTauOutput/{df_}"
df = pd.read_hdf(indf)
#df = df.drop(columns=["total_tau_phi"]) # new
df = df.dropna()
print(df.head(10))


# In[8]:


#hasnan = list(df.isnull().any())
#for i, col in enumerate(df.keys()):
#    print(f"{col}:\t\t {hasnan[i]}")


# In[9]:


#import awkward as ak
#arr = ak.Array((df["rawPNetVSe_tau_1"]).to_numpy())
#ak.to_list(ak.sort(arr))


# In[10]:


print(f"df keys:\n{list(df.keys())}")


# In[11]:


num_keys   = ['pt_tau_1', 'pt_tau_2','eta_tau_1', 'eta_tau_2', 'phi_tau_1', 'phi_tau_2',
              'mass_tau_1', 'mass_tau_2', 'dxy_tau_1', 'dxy_tau_2', 'dz_tau_1', 'dz_tau_2',
              'rawIsodR03_tau_1', 'rawIsodR03_tau_2', 'leadTkPtOverTauPt_tau_1', 
              'leadTkPtOverTauPt_tau_2', 
              'rawPNetVSe_tau_1', 'rawPNetVSe_tau_2',
              'rawPNetVSjet_tau_1', 'rawPNetVSjet_tau_2',
              'rawPNetVSmu_tau_1', 'rawPNetVSmu_tau_2',
              'probDM0PNet_tau_1', 'probDM0PNet_tau_2',
              'probDM1PNet_tau_1', 'probDM1PNet_tau_2',
              'probDM2PNet_tau_1', 'probDM2PNet_tau_2',
              'probDM10PNet_tau_1', 'probDM10PNet_tau_2',
              'probDM11PNet_tau_1', 'probDM11PNet_tau_2',
              'pt_jet_1', 'pt_jet_2', 'pt_jet_3', 'pt_jet_4',
              'eta_jet_1', 'eta_jet_2', 'eta_jet_3', 'eta_jet_4',
              'phi_jet_1', 'phi_jet_2', 'phi_jet_3', 'phi_jet_4',
              'mass_jet_1', 'mass_jet_2', 'mass_jet_3', 'mass_jet_4',
              'btagPNetB_jet_1', 'btagPNetB_jet_2', 'btagPNetB_jet_3', 'btagPNetB_jet_4',
              'npf_tau_1', 'npf_tau_2', 
              'ptfrac_tau_1', 'ptfrac_tau_2', 
              'pf_dphi_pt_frac_tau_1', 'pf_dphi_pt_frac_tau_2', 
              'pf_deta_pt_frac_tau_1', 'pf_deta_pt_frac_tau_2', 
              'pt_MET', 'phi_MET', 'covXX_MET', 'covXY_MET', 'covYY_MET', 'sumEt_MET', 'significance_MET', 
              'nJet', 'nPV', 'HT', 'MT_total']
cat_keys = ['charge_-1_tau_1','charge_1_tau_1',
            'charge_-1_tau_2','charge_1_tau_2',
            'decayModePNet_0_tau_1', 'decayModePNet_1_tau_1', 'decayModePNet_2_tau_1', 
            'decayModePNet_10_tau_1', 'decayModePNet_11_tau_1',
            'decayModePNet_0_tau_2', 'decayModePNet_1_tau_2', 'decayModePNet_2_tau_2', 
            'decayModePNet_10_tau_2', 'decayModePNet_11_tau_2']
target_keys = ['px_gentaunu_1', 'py_gentaunu_1', 'pz_gentaunu_1', 
               'px_gentaunu_2', 'py_gentaunu_2', 'pz_gentaunu_2',
               'h1x', 'h1y', 'h1z',
               'h2x', 'h2y', 'h2z']
extra_keys = ["pt_tau1pi_1","eta_tau1pi_1","phi_tau1pi_1","mass_tau1pi_1",
              "pt_tau2pi_1","eta_tau2pi_1","phi_tau2pi_1","mass_tau2pi_1",
              "pt_tau1pi0_1","eta_tau1pi0_1","phi_tau1pi0_1","mass_tau1pi0_1",
              "pt_tau2pi0_1","eta_tau2pi0_1","phi_tau2pi0_1","mass_tau2pi0_1",
              "phicp", "phi_cp_det"]
keysTokeep = num_keys + cat_keys + target_keys + extra_keys


# In[12]:


df = df[keysTokeep]


# In[13]:


tau1p4 = getp4(df, vars=["pt_tau_1", "eta_tau_1", "phi_tau_1", "mass_tau_1"], LVtype="ptetaphim")
tau2p4 = getp4(df, vars=["pt_tau_2", "eta_tau_2", "phi_tau_2", "mass_tau_2"], LVtype="ptetaphim")

tausp4 = tau1p4 + tau2p4
print(tausp4.mass)
plt.hist(tausp4.mass, bins=100, range=[30,130])
plt.show()
#df["H_vism"] = tausp4.mass
#target_keys = target_keys + ["H_vism"]


# In[14]:


#df_train, df_test = train_test_split(df,  test_size=0.2,  random_state=42, shuffle=True)
df_train, df_test = train_test_split(df,  test_size=0.2,  shuffle=False)


# In[15]:


df_train_x = df_train.drop(columns=target_keys+extra_keys)
df_train_y = df_train[target_keys]

df_test_x = df_test.drop(columns=target_keys+extra_keys)
df_test_y = df_test[target_keys]

#df2_met_raw_phi_train = df_train_x[["met_phi_raw"]]
#df2_met_raw_phi_test  = df_test_x[["met_phi_raw"]]
#df_train_x = df_train_x.drop(columns=["met_phi_raw"])
#df_test_x = df_test_x.drop(columns=["met_phi_raw"])

feats = list(df_train_x.keys())


# In[16]:


if evaluate:
    df = pd.concat([df_train, df_test], axis=0)


# In[17]:


df_train_x


# In[18]:


df_train_y


# In[19]:


df_train_x.keys()


# ## Data engineering for LBN 

# In[20]:


def dataset_LBN():    
    import vector
    df_train_x_copy = df_train_x.copy()
    df_test_x_copy = df_test_x.copy()

    def getp4(df, vars=[]):
        #for var in vars:
        #    if var not in list(df.keys()):
        #        df[var] = 0.0
        p4 = vector.array({"pt": df[vars[0]].to_numpy(),
                           "eta": df[vars[1]].to_numpy() if vars[1] in list(df.keys()) else np.zeros_like(df[vars[0]].to_numpy()),
                           "phi": df[vars[2]].to_numpy() if vars[2] in list(df.keys()) else np.zeros_like(df[vars[0]].to_numpy()),
                           "M": df[vars[3]].to_numpy() if vars[3] in list(df.keys()) else np.zeros_like(df[vars[0]].to_numpy())})

        return p4

    def setp4(df, vars=[]):
        p4 = getp4(df, vars=vars)
        obj = vars[0].split('_')[1]
        num = vars[0].split('_')[-1]
        key = f'{obj}_{num}' if len(vars[0].split('_')) > 2 else f'{obj}'

        df[f'px_{key}'] = p4.px
        df[f'py_{key}'] = p4.py
        df[f'pz_{key}'] = p4.pz
        df[f'E_{key}']  = p4.E



    Basics = [['pt_tau_1','eta_tau_1','phi_tau_1','mass_tau_1'],
              ['pt_tau_2','eta_tau_2','phi_tau_2','mass_tau_2'],
              ['pt_jet_1','eta_jet_1','phi_jet_1','mass_jet_1'],
              ['pt_jet_2','eta_jet_2','phi_jet_2','mass_jet_2'],
              ['pt_jet_3','eta_jet_3','phi_jet_3','mass_jet_3'],
              ['pt_jet_4','eta_jet_4','phi_jet_4','mass_jet_4'],
              ['pt_MET','eta_MET','phi_MET','mass_MET']]

    for varlist in Basics:
        setp4(df_train_x_copy, vars=varlist)
        setp4(df_test_x_copy, vars=varlist)


    DNNsToDrop = np.array(Basics).reshape(-1)    
    #print(DNNsToDrop)

    LBNsToKeep = np.char.replace(DNNsToDrop, "pt", "px")
    LBNsToKeep = np.char.replace(LBNsToKeep, "eta", "py")
    LBNsToKeep = np.char.replace(LBNsToKeep, "phi", "pz")
    LBNsToKeep = np.char.replace(LBNsToKeep, "mass", "E")
    LBNsToKeep = list(LBNsToKeep)
    #print(LBNsToKeep)

    #df_train_x_DNN = df_train_x_copy.drop(columns=list(DNNsToDrop)+LBNsToKeep)
    #df_test_x_DNN = df_test_x_copy.drop(columns=list(DNNsToDrop)+LBNsToKeep)
    df_train_x_DNN = df_train_x_copy.drop(columns=LBNsToKeep)
    df_test_x_DNN = df_test_x_copy.drop(columns=LBNsToKeep)

    df_train_x_LBN = df_train_x_copy[LBNsToKeep]
    df_test_x_LBN  = df_test_x_copy[LBNsToKeep]


    ##df_train_x_DNN
    ##df_train_x_DNN
    ##df_train_x_LBN
    ##df_test_x_LBN
    
    #cols_cat = [x for x in list(df_train_x_DNN.keys()) if x.split('_')[0] == 'cat']
    #cols_num = [x for x in list(df_train_x_DNN.keys()) if x.split('_')[0] != 'cat']
    
    x_train_DNN_numerical = df_train_x_DNN[num_keys].to_numpy()
    x_train_DNN_categorical = df_train_x_DNN[cat_keys].to_numpy()
    x_test_DNN_numerical = df_test_x_DNN[num_keys].to_numpy()
    x_test_DNN_categorical = df_test_x_DNN[cat_keys].to_numpy()

    #x_train_DNN = df_train_x_DNN.to_numpy()
    #x_test_DNN = df_test_x_DNN.to_numpy()

    ### ----- Scaling ----- ###
    scaler1 = preprocessing.StandardScaler()
    x_train_DNN_numerical = scaler1.fit_transform(x_train_DNN_numerical)
    x_test_DNN_numerical  = scaler1.transform(x_test_DNN_numerical)

    x_train_DNN = np.concatenate([x_train_DNN_numerical, x_train_DNN_categorical], axis=1)
    x_test_DNN  = np.concatenate([x_test_DNN_numerical, x_test_DNN_categorical], axis=1)
    ### ------------------- ###
    
    #x_train_DNN = df_train_x_DNN.to_numpy()
    #x_test_DNN  = df_test_x_DNN.to_numpy()
    
    x_train_LBN_ = df_train_x_LBN.to_numpy()
    x_test_LBN_ = df_test_x_LBN.to_numpy()

    ### ----- Scaling ----- ###
    #scaler2 = preprocessing.StandardScaler()
    #x_train_LBN_ = scaler2.fit_transform(x_train_LBN_)
    #x_test_LBN_  = scaler2.transform(x_test_LBN_)
    ### ------------------- ###
    
    x_train_LBN = x_train_LBN_.reshape(-1, x_train_LBN_.shape[-1]//4, 4)
    x_test_LBN = x_test_LBN_.reshape(-1, x_test_LBN_.shape[-1]//4, 4)

    print(x_train_LBN.shape, x_train_DNN.shape)
    print(x_train_LBN.shape, x_test_DNN.shape)
    
    """
    plt.figure(figsize=(20,20))
    for idx, key in enumerate(list(df_train_x_LBN.keys())):
        idx=idx+1
        ax = plt.subplot(7,4,idx)
        arr = x_train_LBN_[:,(idx-1):idx].reshape(-1)
        #print(arr)
        ax.hist(arr, 50, range=[np.min(arr), np.max(arr)], log=True)
        ax.set_title(f"{key}")
    plt.tight_layout()
    """

    feats_DNN = list(df_train_x_DNN.keys())
    
    dict = {'x_train': np.concatenate([x_train_DNN, x_train_LBN_], axis=1),
            'x_test': np.concatenate([x_test_DNN, x_test_LBN_], axis=1),
            'x_train_LBN_keys': list(df_train_x_LBN.keys()),
            'x_train_DNN_keys': list(df_train_x_DNN.keys()),
            'x_train_LBN': x_train_LBN,
            'x_test_LBN': x_test_LBN,
            'x_train_DNN': x_train_DNN,
            'x_test_DNN': x_test_DNN}
    return dict


# In[21]:


LBN_dict = dataset_LBN()

feats_name_LBN = LBN_dict['x_train_LBN_keys']
feats_name_DNN = LBN_dict['x_train_DNN_keys']

x_train_LBN =  LBN_dict['x_train_LBN']
x_train_DNN =  LBN_dict['x_train_DNN']
x_test_LBN  =  LBN_dict['x_test_LBN']
x_test_DNN  =  LBN_dict['x_test_DNN']


# In[22]:


len(feats_name_DNN), len(feats_name_LBN)


# In[23]:


feats = feats_name_LBN+feats_name_DNN if "LBN" in tag_ else feats
#feats = feats_name_DNN if tag_=="LBN" else feats
len(feats)


# In[ ]:





# In[24]:


x_train = df_train_x.to_numpy()
y_train = df_train_y.to_numpy()

x_test = df_test_x.to_numpy()
y_test = df_test_y.to_numpy()


# In[25]:


x_train.shape


# In[26]:


#x_train[:,-12:]
x_train.shape, y_train.shape


# In[27]:


features = list(df_train_x.keys())
print(len(features))
targets = list(df_train_y.keys())
print(len(targets))


# In[28]:


"""
scaler = preprocessing.StandardScaler()
x_train_ = scaler.fit_transform(x_train[:, :-12])
x_test_  = scaler.transform(x_test[:, :-12])

x_train = np.concatenate([x_train_, x_train[:,-12:]], axis=1)
x_test  = np.concatenate([x_test_, x_test[:,-12:]], axis=1)
"""


# ## Data batching 

# In[29]:


#scaler = preprocessing.StandardScaler()
#y_train = scaler.fit_transform(y_train)
#y_test  = scaler.transform(y_test)


# In[ ]:





# In[ ]:





# ## Model subclassing 

# In[ ]:





# In[30]:


#model = DNNModel()
y_train.shape


# In[31]:


plt.hist(y_train[:,1], bins=40, range=[-40,40])
plt.show()


# In[32]:


px_temp = y_train[:,1]
px_temp


# In[33]:


np.mean(px_temp), np.std(px_temp)


# In[34]:


#wt = np.where(np.abs(px_temp) < np.std(px_temp), 1+np.abs(np.log(1/(px_temp**2))), 1.0)
#wt


# In[35]:


wt = 1+np.abs(np.log(1/(px_temp**2)))
wt


# In[36]:


plt.scatter(px_temp, wt)


# In[37]:


plt.hist(wt, bins=100, range=[1,20], log=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


"""
plt.figure(figsize=(40,50))
for idx, key in enumerate(features):
    idx=idx+1
    ax = plt.subplot(13,6,idx)
    arr = x_train[:,(idx-1):idx].reshape(-1)
    #print(arr)
    ax.hist(arr, 50, range=[np.min(arr), np.max(arr)], log=True)
    ax.set_title(f"{key}")
plt.tight_layout()
"""


# In[39]:


#feats


# In[40]:


inputs_all = []
for feat in feats:
    #print(feat)
    input_layer = Input(shape=(1,), name=feat)
    inputs_all.append(input_layer)
#print(inputs_all)
#print(len(inputs_all))
#print(inputs_all[1].shape)

all_features = tensorflow.keras.layers.concatenate(inputs_all,axis=-1,name="Features")


# In[41]:


all_features


# In[ ]:





# In[42]:


all_features[0][-1]


# In[43]:


def buildmodel_DNN(input_layers, features):
    print("Building a DNN model")
    x = Dense(512, input_dim=features.shape[-1], activation='selu',
              kernel_regularizer=regularizers.l2(0.001))(features)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(7)(x)

    model_inputs = [input_layers]
    model = Model(inputs=model_inputs, outputs=[x])

    return model


# In[44]:


def identity_block(X, layerinfo):
    X_shortcut = X
    for i, nodes in enumerate(layerinfo):
        X = Dense(nodes, kernel_regularizer=regularizers.l2(0.001))(X)
        X = Activation('selu')(X)
        X = BatchNormalization()(X)

    X = Add()([X, X_shortcut])
    X = Activation('selu')(X)

    return X

def conv_block(X, layerinfo):
    X_shortcut = X
    for i, nodes in enumerate(layerinfo):
        X = Dense(nodes, activation='selu', kernel_regularizer=regularizers.l2(0.0001))(X)
        #X = Activation('selu')(X)
        #X = BatchNormalization()(X)

    #X_shortcut = Dense(layerinfo[-1])(X)
    X_shortcut = Dense(layerinfo[-1])(X_shortcut)
    #X_shortcut = BatchNormalization()(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('selu')(X)
    X = BatchNormalization()(X)

    return X

def buildmodel_Resnet(input_layers, features):
    print("Building a SkipNetwork DNN model")
    X = Dense(512, input_dim=features.shape[-1], activation='selu',
              kernel_regularizer=regularizers.l2(0.0001))(features)
    X = BatchNormalization()(X)
    X = Dropout(0.1)(X)

    X = conv_block(X, [512, 512, 512])
    X = Dropout(0.1)(X)
    X = conv_block(X, [256, 256, 256])
    X = Dropout(0.1)(X)
    X = conv_block(X, [128, 128, 128])
    X = Dropout(0.1)(X)
    X = conv_block(X, [64, 64, 64])
    X = Dropout(0.1)(X)
    X = Dense(7)(X)


    model_inputs = [input_layers]
    print(f"model_inputs: {model_inputs}")
    model = Model(inputs=model_inputs, outputs=[X])

    return model


# In[45]:


def buildmodel_LBN(inputs_all, x_train_LBN, all_features):
    print("Building a LBN model")
    input_lbn_Layer = tensorflow.keras.Input(shape=x_train_LBN.shape[1:],name='LBN_inputs')
    print(f'type & shape of lbn input: {type(input_lbn_Layer)}, {input_lbn_Layer.shape}')

    lbn_layer = LBNLayer(x_train_LBN.shape[1:],
                         n_particles = 7,
                         boost_mode  = LBN.PAIRS,
                         features    = ["E", "px", "py", "pz", "pt", "p", "phi","eta", "m", "pair_cos"],
                         #features    = ["E","px","py","pz","pt","p","m","phi","eta","beta","gamma","pair_cos","pair_dr","pair_ds","pair_dy"],
                         name='LBN')(input_lbn_Layer)
    print(f'Type & shape of LBN layer : {type(lbn_layer)}')

    batchnorm = tensorflow.keras.layers.BatchNormalization(name='batchnorm')(lbn_layer)
    print(f'type & shape of batchnorm : {type(batchnorm)}, {batchnorm.shape}')

    concat = tensorflow.keras.layers.Concatenate(axis=-1)([all_features, batchnorm])
    print(f'Concatenated shape : {concat.shape}')
    print(f'Concatenated type  : {type(concat)}')


    x = Dense(512, activation='selu', kernel_regularizer=regularizers.l2(0.001))(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(8)(x)

    model_inputs = [inputs_all]
    model_inputs.append(input_lbn_Layer)
    print(model_inputs)
    model = Model(inputs=model_inputs, outputs=[x])

    return model


# In[46]:


def buildmodel_LBN_ResNet(inputs_all, x_train_LBN, all_features):
    print("Building a LBN with skip connection model") 
    input_lbn_Layer = tensorflow.keras.Input(shape=x_train_LBN.shape[1:],name='LBN_inputs')
    print(f'type & shape of lbn input: {type(input_lbn_Layer)}, {input_lbn_Layer.shape}')

    lbn_layer = LBNLayer(x_train_LBN.shape[1:],
                         n_particles = 5,
                         boost_mode  = LBN.PAIRS,
                         features    = ["E", "px", "py", "pz", "pt", "eta", "phi", "p", "m", "pair_cos","pair_dr"],
                         #features    = ["E","px","py","pz","pt","p","m","phi","eta","beta","gamma","pair_cos","pair_dr","pair_ds","pair_dy"],
                         name='LBN')(input_lbn_Layer)
    print(f'Type & shape of LBN layer : {type(lbn_layer)}')

    batchnorm = tensorflow.keras.layers.BatchNormalization(name='batchnorm')(lbn_layer)
    print(f'type & shape of batchnorm : {type(batchnorm)}, {batchnorm.shape}')

    concat = tensorflow.keras.layers.Concatenate(axis=-1)([all_features, batchnorm])
    print(f'Concatenated shape : {concat.shape}')
    print(f'Concatenated type  : {type(concat)}')


    #x = Dense(512, activation='selu', kernel_regularizer=regularizers.l2(0.001))(concat)
    #x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)

    #x = conv_block(concat, [1028, 1028])
    #x = Dropout(0.1)(x)
    x = conv_block(concat, [512, 512, 512])
    x = Dropout(0.1)(x)
    x = conv_block(x, [256, 256, 256])
    x = Dropout(0.1)(x)
    x = conv_block(x, [256, 256, 256])
    x = Dropout(0.1)(x)
    x = conv_block(x, [128, 128, 128])
    x = Dropout(0.1)(x)
    #x = Dense(8)(x)
    x = Dense(12)(x)

    model_inputs = [inputs_all]
    model_inputs.append(input_lbn_Layer)
    print(model_inputs)
    model = Model(inputs=model_inputs, outputs=[x])

    return model



def getmodel(input_layers=inputs_all, x_train_LBN=x_train_LBN, features=all_features, tag=tag_):
    model = None
    tag_split = tag_.split('_')[0]
    if tag_split=="DNN":
        model = buildmodel_DNN(input_layers, features)
    elif tag_split=="LBN":
        model = buildmodel_LBN(input_layers, x_train_LBN, features)
    elif tag_split=="ResNet":
        model = buildmodel_Resnet(input_layers, features)
    elif tag_split=="LBNResNet":
        model = buildmodel_LBN_ResNet(input_layers, x_train_LBN, features)
    else:
        print("Run DNN as no tag mentioned")
        model = buildmodel_DNN(input_layers, features)

    return model


# In[48]:


model = getmodel(tag=tag_)
model.summary()


# In[49]:


# Compile and fit
#set early stopping monitor so the model stops training when it won't improve anymore                                                                     
#early_stopping_monitor = EarlyStopping(patience=3)                                                                                                       
# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd                                              
custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    #monitor='loss',
    #patience=int(nEpoch/10),
    patience=20,                                                                                                                                         
    min_delta=0.0001,
    verbose=1,
    restore_best_weights=True,
    #restore_best_weights=False,                                                                                                                          
    mode='min'
    #mode='max'                                                                                                                                           
)
#https://keras.io/api/callbacks/reduce_lr_on_plateau/                                                                                                     
custom_ReduceLROnPlateau = ReduceLROnPlateau(
    monitor="val_loss",
    #monitor="loss",                                                                                                                  
    factor=0.5, #0.1
    #patience=5,                                                                                                                                          
    patience=10,
    verbose=1,
    mode="min",
    #mode="max",                                                                                                                                          
    cooldown=0,                                                                                                                                          
    #cooldown=5,
    #min_lr=0
    min_lr=1e-10,                                                                                                                                         
    min_delta=0.0001                                                                                                                                      
)
#custom loss-history with batch
#https://github.com/cp3-llbb/HHbbWWAnalysis/blob/master/MachineLearning/HHMachineLearning/Model.py#L47
class LossHistory(Callback):
    """ Records the history of the training per epoch and per batch """
    def on_train_begin(self, logs={}):
        self.epochs  = defaultdict(list) 
        self.batches = defaultdict(list) 
        self.pre_batch = 0

    def on_batch_end(self, batch, logs={}):
        self.batches['batch'].append(batch+self.pre_batch)
        for key,val in logs.items():
            self.batches[key].append(val)
        self.batches['lr'].append(tensorflow.keras.backend.eval(self.model.optimizer.lr))
        #loss = logs.get('loss')
        #print(f'\nBatch : {batch} with average loss : {loss}')

    def on_epoch_end(self, epoch, logs={}):
        self.epochs['epoch'].append(epoch)
        for key,val in logs.items():
            self.epochs[key].append(val)
        #self.epochs['lr'].append(tensorflow.keras.backend.eval(self.model.optimizer.lr))
        self.pre_batch = self.batches['batch'][-1] 


# In[50]:


#print(x_train.shape)
#fit_inputs = np.hsplit(x_train,x_train.shape[1])
#print(len(fit_inputs))

x_train = LBN_dict['x_train'] if "LBN" in tag_ else x_train
x_test  = LBN_dict['x_test'] if "LBN" in tag_ else x_test
#x_train = x_train_DNN if tag_=="LBN" else x_train
#x_test  = x_test_DNN if tag_=="LBN" else x_test


print(x_train.shape)
fit_inputs = np.hsplit(x_train, x_train.shape[1])
print(len(fit_inputs))
if "LBN" in tag_:
    fit_inputs.append(x_train_LBN)

print(len(fit_inputs))
print(fit_inputs[0].shape)
print(fit_inputs[-1].shape)


fit_tests = np.hsplit(x_test, x_test.shape[1])
if "LBN" in tag_:
    fit_tests.append(x_test_LBN)



def custom_loss():
    def loss(y_true, y_pred):
        print(f"y_true: {y_true}")
        print(f"y_pred: {y_pred}")
        #lbn_input = np.array(x_input[-1])
        #print(f"x_input: {lbn_input.shape}")
        #print("Check y in detail")
        print(y_true[:,:1], y_pred[:,:1])
        print(y_true[:,1:2], y_pred[:,1:2])
        print(y_true[:,2:3], y_pred[:,2:3])
        #x = tf.convert_to_tensor(x_input)
        #print(f'x: {x[:,:1]}')
        """
        tau1_p4 = x_lbn_input[:y_pred.shape[0],0:1,:].reshape(y_pred.shape[0],4)
        tau2_p4 = x_lbn_input[:y_pred.shape[0],1:2,:].reshape(y_pred.shape[0],4)
        #print(f"tau1 : {x_lbn_input[:,:1,:].reshape(x_lbn_input.shape[0],4)}")
        lvadd_tau1nu1_pred = tf.add(y_pred[:,:4], tau1_p4)
        lvadd_tau1nu1_true = tf.add(y_true[:,:4], tau1_p4)
        print(f"lvadd_tau1nu1_pred: {lvadd_tau1nu1_pred}")

        lvadd_tau2nu2_pred = tf.add(y_pred[:,4:8], tau2_p4)
        lvadd_tau2nu2_true = tf.add(y_true[:,4:8], tau2_p4)

        energy_tau1nu1_pred = lvadd_tau1nu1_pred[:, 0]
        momentum_tau1nu1_pred = tf.norm(lvadd_tau1nu1_pred[:, 1:], axis=1)
        mass_tau1nu1_pred = tf.sqrt(energy_tau1nu1_pred ** 2 - momentum_tau1nu1_pred ** 2)

        energy_tau1nu1_true = lvadd_tau1nu1_true[:, 0]
        momentum_tau1nu1_true = tf.norm(lvadd_tau1nu1_true[:, 1:], axis=1)
        mass_tau1nu1_true = tf.sqrt(energy_tau1nu1_true ** 2 - momentum_tau1nu1_true ** 2)

        energy_tau2nu2_pred = lvadd_tau2nu2_pred[:, 0]
        momentum_tau2nu2_pred = tf.norm(lvadd_tau2nu2_pred[:, 1:], axis=1)
        mass_tau2nu2_pred = tf.sqrt(energy_tau2nu2_pred ** 2 - momentum_tau2nu2_pred ** 2)

        energy_tau2nu2_true = lvadd_tau2nu2_true[:, 0]
        momentum_tau2nu2_true = tf.norm(lvadd_tau2nu2_true[:, 1:], axis=1)
        mass_tau2nu2_true = tf.sqrt(energy_tau2nu2_true ** 2 - momentum_tau2nu2_true ** 2)


        mass = tf.reduce_mean(tf.add(tf.square(mass_tau1nu1_true - mass_tau1nu1_pred), tf.square(mass_tau2nu2_true - mass_tau2nu2_pred)))

        #tau_mass_pred = tf.reduce_mean(tf.add(mass_tau1nu1_pred, mass_tau2nu2_pred))
        #tau_mass_true = tf.reduce_mean(tf.add(mass_tau1nu1_true, mass_tau2nu2_true))

        #massterm = tf.square(tau_mass_pred - tau_mass_true)

        print(f"mass: {mass}")
        """
        se = tf.square(y_true[:,:-1]-y_pred[:,:-1])
        #mse = tf.reduce_mean(se)
        mass = tf.square(y_true[:,-1:] - y_pred[:,-1:])
        #mass = tf.reduce_mean(mass)
        #mse = tf.cast(mse, tf.float64)
        # print(f"mse: {mse}")
        return tf.reduce_mean(tf.add(se,mass))
        #return tf.add(mse, mass)
    return loss


def huber_loss(y_true, y_pred, threshold):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold     
    small_error_loss = tf.square(error) / 2     
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    loss = tf.where(is_small_error, small_error_loss, big_error_loss)
    #mean_loss = tf.reduce_mean(loss)
    #print(f"Loss: {loss}")
    return loss

def custom_loss_2(threshold=1.0, beta=2.0):
    def huber_loss_mean(y_true, y_pred):
        huber_loss_val = tf.reduce_mean(huber_loss(y_true[:,:-1], y_pred[:,:-1], threshold), 1)
        print(f"huber_loss_val: {huber_loss_val}")
        mass_huber_loss_val = huber_loss(y_true[:,-1], y_pred[:,-1], threshold)
        print(f"mass_huber_loss_val: {mass_huber_loss_val}")
        loss = tf.reduce_mean(tf.add(huber_loss_val, beta*mass_huber_loss_val))  
        return loss
    return huber_loss_mean

def custom_loss_3(alpha=1.0, beta=1.0):
    def comp_loss(y_true, y_pred):
        #mae_loss_val = tf.reduce_sum(tf.abs(y_true[:,:-1] - y_pred[:,:-1]))
        mae_loss_val = alpha*tf.abs(y_true[:,:-1] - y_pred[:,:-1])
        #mae_loss_val = tf.cumsum(mae_loss_val, 1)
        print(f"mae_loss_val: {mae_loss_val}")
        #mass_loss_val = tf.reduce_sum(beta*((y_pred[:,-1]/y_true[:,-1]) - 1))
        mass_loss_val = tf.abs(beta*((y_pred[:,-1]/y_true[:,-1]) - 1))
        print(f"mass_loss_val: {mass_loss_val}")
        loss = tf.concat([mae_loss_val, tf.reshape(mass_loss_val, [-1,1])], 1)
        #loss = tf.reduce_sum(loss)
        loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
        return loss
    return comp_loss


# In[64]:


t1 = [[1, 2, 3], [4, 5, 6]]
t1 = tf.reshape(t1, [-1])
t2 = tf.reshape(t1, [-1,1])
t2


# In[65]:


loss_history = LossHistory()
#y_pred = model.evaluate()
#tau1phi = df_train_x["tau_1_phi"].to_numpy()
#tau1phi = np.abs(tau1phi)
model.compile(loss=tensorflow.keras.losses.Huber(), #custom_loss_3(), custom_loss(fit_inputs), custom_loss(x_train_LBN), custom_loss_1(df_train_x), tensorflow.keras.losses.Huber(), custom_loss_1(tau1phi),
              optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.01),
              metrics=[tensorflow.keras.metrics.R2Score(),
                       tensorflow.keras.metrics.MeanAbsoluteError(),
                       tensorflow.keras.metrics.MeanAbsolutePercentageError()])


log_dir = f"{tag_}/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

datetime_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"{tagdir}/training_{datetime_tag}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callback_list = [#custom_early_stopping,
                 custom_ReduceLROnPlateau,
                 loss_history,
                 tensorboard_callback,
                 cp_callback]


# In[ ]:


history = None
if train:
    history = model.fit(fit_inputs,
                        y_train,
                        epochs=200,
                        batch_size=1024,
                        validation_split=0.2,
                        #sample_weight=wt,
                        verbose=1,
                        use_multiprocessing=True,
                        callbacks=callback_list)
    PlotHistory(loss_history, path=tagdir)
else:
    checkpoint_path = f"{tagdir}/training_20240404-120039/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = getmodel(tag=tag_)
    model.load_weights(checkpoint_path)
    print(f"Loaded model from checkpoint path: {checkpoint_path}")


# In[ ]:


"""
loadmodel = None
if train: 
    PlotHistory(loss_history, path=tagdir)
else:
    checkpoint_path = f"{tagdir}/training_20231102-154447/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    loadmodel = getmodel(tag=tag_)
    loadmodel.load_weights(checkpoint_path)
    print(f"Loaded model from checkpoint path: {checkpoint_path}")
"""


# In[ ]:


#if not train:
#    model = loadmodel
target_reg_val_train   = model.predict(fit_inputs)
target_val_train = y_train
target_reg_val_test    = model.predict(fit_tests)
target_val_test  = y_test


# In[ ]:


target_regr = None
target_true = None
if evaluate:
    target_regr = np.concatenate((target_reg_val_train, target_reg_val_test), axis=0)
    target_true = np.concatenate((target_val_train, target_val_test), axis=0)


# In[ ]:


plt.figure(figsize=(24,14))
for idx, key in enumerate(targets):
    idx=idx+1
    ax = plt.subplot(5,3,idx)
    arr_true = target_val_test[:,(idx-1):idx].reshape(-1)
    arr_pred = target_reg_val_test[:,(idx-1):idx].reshape(-1)

    #print(arr)
    minmax = [-50,50] if idx < 7 else [-1,1]
    ax.hist(arr_true, 100, histtype="stepfilled", alpha=0.7, label='True', log=False, density=True)
    ax.hist(arr_pred, 100, histtype="stepfilled", alpha=0.7, label='Regressed [Test]', log=False, density=True)
    ax.set_title(f"{key}")
    ax.set_xlabel(f"""{key.split('_')[-1]}""")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_test.png'), dpi=300)


# In[ ]:


"""
plt.figure(figsize=(20,16))
for idx, key in enumerate(targets):
    idx=idx+1
    ax = plt.subplot(4,3,idx)
    arr_true = target_val_test[:,(idx-1):idx].reshape(-1)
    arr_pred = target_reg_val_test[:,(idx-1):idx].reshape(-1)

    #print(arr)
    minmax = [0,50] #if idx%4 == 0 else [0,6.5]
    ax.hist(np.abs(arr_true), 50, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
    ax.hist(np.abs(arr_pred), 50, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed [Test]')
    ax.set_title(f"{key}")
    ax.set_xlabel(f"{key.split('_')[-1]}")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_test_abs.png'), dpi=300)
"""


# In[ ]:


plt.figure(figsize=(24,14))
for idx, key in enumerate(targets):
    idx=idx+1
    ax = plt.subplot(5,3,idx)
    arr_true = target_val_train[:,(idx-1):idx].reshape(-1)
    arr_pred = target_reg_val_train[:,(idx-1):idx].reshape(-1)

    #print(arr)
    #minmax = [-50,50] if np.min(arr_true) < 0 else [0,1]
    minmax = [-50,50] if idx < 7 else [-1,1]
    ax.hist(arr_true, 100, histtype="stepfilled", alpha=0.7, label='True', 
            log=False, density=True)
    ax.hist(arr_pred, 100, histtype="stepfilled", alpha=0.7, label='Regressed [Train]', 
            log=False, density=True)
    ax.set_title(f"{key}")
    ax.set_xlabel(f'''{key.split('_')[-1]}''')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_train.png'), dpi=300)


# In[ ]:


"""
plt.figure(figsize=(20,16))
for idx, key in enumerate(targets):
    idx=idx+1
    ax = plt.subplot(2,3,idx)
    arr_true = target_val_train[:,(idx-1):idx].reshape(-1)
    arr_pred = target_reg_val_train[:,(idx-1):idx].reshape(-1)

    #print(arr)
    minmax = [0,50] #if idx%4 == 0 else [0,50]
    ax.hist(np.abs(arr_true), 50, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
    ax.hist(np.abs(arr_pred), 50, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed [Train]')
    ax.set_title(f"{key}")
    ax.set_xlabel(f"{key.split('_')[-1]}")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_train_abs.png'), dpi=300)
"""


# In[ ]:


true_nu1_p4 = vector.array({"px": target_val_test[:,:1].reshape(-1),
                            "py": target_val_test[:,1:2].reshape(-1),
                            "pz": target_val_test[:,2:3].reshape(-1),
                            #"E": target_val_test[:,3:4]})
                            "E": np.sqrt(target_val_test[:,:1]**2 + target_val_test[:,1:2]**2 + target_val_test[:,2:3]**2).reshape(-1)})
true_nu2_p4 = vector.array({"px": target_val_test[:,3:4].reshape(-1),
                            "py": target_val_test[:,4:5].reshape(-1),
                            "pz": target_val_test[:,5:6].reshape(-1),
                            #"E": target_val_test[:,7:8]})
                            "E": np.sqrt(target_val_test[:,3:4]**2 + target_val_test[:,4:5]**2 + target_val_test[:,5:6]**2).reshape(-1)})


# In[ ]:


target_val_test[:,:1].reshape(-1).shape


# In[ ]:


true_nu1_p4.phi


# In[ ]:


#rawmetphi_test = df2_met_raw_phi_test["met_phi_raw"].to_numpy().reshape(-1)


# In[ ]:


## add met phi
#true_nu1_p4 = vector.array({"pt": true_nu1_p4.pt,
#                            "eta": true_nu1_p4.eta,
#                            "phi": true_nu1_p4.phi + rawmetphi_test,
#                            "M": np.zeros_like(true_nu1_p4.pt)})
#true_nu2_p4 = vector.array({"pt": true_nu2_p4.pt,
#                            "eta": true_nu2_p4.eta,
#                            "phi": true_nu2_p4.phi + rawmetphi_test,
#                            "M": np.zeros_like(true_nu2_p4.pt)})


# In[ ]:


true_nu1_p4


# In[ ]:


regr_nu1_p4 = vector.array({"px": target_reg_val_test[:,:1].reshape(-1),
                            "py": target_reg_val_test[:,1:2].reshape(-1),
                            "pz": target_reg_val_test[:,2:3].reshape(-1),
                            #"E": target_val_test[:,3:4]})
                            "E": np.sqrt(target_reg_val_test[:,:1]**2 + target_reg_val_test[:,1:2]**2 + target_reg_val_test[:,2:3]**2).reshape(-1)})
regr_nu2_p4 = vector.array({"px": target_reg_val_test[:,3:4].reshape(-1),
                            "py": target_reg_val_test[:,4:5].reshape(-1),
                            "pz": target_reg_val_test[:,5:6].reshape(-1),
                            #"E": target_val_test[:,7:8]})
                            "E": np.sqrt(target_reg_val_test[:,3:4]**2 + target_reg_val_test[:,4:5]**2 + target_reg_val_test[:,5:6]**2).reshape(-1)})

"""
regr_nu1_p4 = vector.array({"px": target_reg_val_test[:,:1],
                            "py": target_reg_val_test[:,1:2],
                            "pz": target_reg_val_test[:,2:3],
                            "E": target_reg_val_test[:,3:4]})
regr_nu2_p4 = vector.array({"px": target_reg_val_test[:,4:5],
                            "py": target_reg_val_test[:,5:6],
                            "pz": target_reg_val_test[:,6:7],
                            "E": target_reg_val_test[:,7:8]})
"""


# In[ ]:


## add met phi
#regr_nu1_p4 = vector.array({"pt": regr_nu1_p4.pt,
#                            "eta": regr_nu1_p4.eta,
#                            "phi": regr_nu1_p4.phi + rawmetphi_test,
#                            "M": np.zeros_like(regr_nu1_p4.pt)})
#regr_nu2_p4 = vector.array({"pt": regr_nu2_p4.pt,
#                            "eta": regr_nu2_p4.eta,
#                            "phi": regr_nu2_p4.phi + rawmetphi_test,
#                            "M": np.zeros_like(regr_nu2_p4.pt)})


# In[ ]:


true_nu1_p4, regr_nu1_p4


# In[ ]:


nu1_dict = {
    'nu1_pt_true': true_nu1_p4.pt,
    'nu1_pt_regr': regr_nu1_p4.pt,
    'nu1_eta_true': true_nu1_p4.eta,
    'nu1_eta_regr': regr_nu1_p4.eta,
    'nu1_phi_true': true_nu1_p4.phi,
    'nu1_phi_regr': regr_nu1_p4.phi,
    'nu1_p_true': true_nu1_p4.p,
    'nu1_p_regr': regr_nu1_p4.p,
}
nu2_dict = {
    'nu2_pt_true': true_nu2_p4.pt,
    'nu2_pt_regr': regr_nu2_p4.pt,
    'nu2_eta_true': true_nu2_p4.eta,
    'nu2_eta_regr': regr_nu2_p4.eta,
    'nu2_phi_true': true_nu2_p4.phi,
    'nu2_phi_regr': regr_nu2_p4.phi,
    'nu2_p_true': true_nu2_p4.p,
    'nu2_p_regr': regr_nu2_p4.p,
}
items = ['pt', 'eta', 'phi', 'p']


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,10))
for idx, key in enumerate(items):
    idx=idx+1
    ax = plt.subplot(2,2,idx)
    arr_true = nu1_dict[f'nu1_{key}_true'].reshape(-1)
    arr_pred = nu1_dict[f'nu1_{key}_regr'].reshape(-1)

    #print(arr)
    minmax = [-3.2,3.2] if np.min(arr_true) < 0 else [0,60]
    ax.hist(arr_true, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
    ax.hist(arr_pred, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed')
    ax.set_title(f"{key}")
    ax.set_xlabel(f"""{key.split('_')[-1]}""")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_nu_1.png'), dpi=300)


# In[ ]:


plt.figure(figsize=(16,10))
for idx, key in enumerate(items):
    idx=idx+1
    ax = plt.subplot(2,2,idx)
    arr_true = nu2_dict[f'nu2_{key}_true'].reshape(-1)
    arr_pred = nu2_dict[f'nu2_{key}_regr'].reshape(-1)

    #print(arr)
    minmax = [-3.2,3.2] if np.min(arr_true) < 0 else [0,60]
    ax.hist(arr_true, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
    ax.hist(arr_pred, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed')
    ax.set_title(f"{key}")
    ax.set_xlabel(f"""{key.split('_')[-1]}""")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'output_nu_2.png'), dpi=300)


# In[ ]:


phitemp1 = (nu1_dict['nu1_phi_true'] - nu1_dict['nu1_phi_regr'])/nu1_dict['nu1_phi_true']
phitemp2 = (nu2_dict['nu2_phi_true'] - nu2_dict['nu2_phi_regr'])/nu2_dict['nu2_phi_true']

etatemp1 = (nu1_dict['nu1_eta_true'] - nu1_dict['nu1_eta_regr'])/nu1_dict['nu1_eta_true']
etatemp2 = (nu2_dict['nu2_eta_true'] - nu2_dict['nu2_eta_regr'])/nu2_dict['nu2_eta_true']

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].hist(phitemp1, bins=100, range=[-1,1], histtype="stepfilled", alpha=0.7, log=False, label= "nu1_phires: true - pred")
ax[0].hist(phitemp2, bins=100, range=[-1,1], histtype="stepfilled", alpha=0.7, log=False, label= "nu2_phires: true - pred")
ax[0].legend()

ax[1].hist(etatemp1, bins=100, range=[-1,1], histtype="stepfilled", alpha=0.7, log=False, label= "nu1_etares: true - pred")
ax[1].hist(etatemp2, bins=100, range=[-1,1], histtype="stepfilled", alpha=0.7, log=False, label= "nu2_etares: true - pred")
ax[1].legend()

plt.tight_layout()
#plt.savefig(os.path.join(tagdir,"tau_nu_pt.png"), dpi=300)
#plt.show()
#plt.xlabel("nu1 (true_phi - pred_phi) / true_phi")
#plt.show()
plt.savefig(os.path.join(tagdir,'nu_eta_phi_res.png'), dpi=300)


# In[ ]:


phitemp1 = (nu1_dict['nu1_pt_true'] - nu1_dict['nu1_pt_regr'])/nu1_dict['nu1_pt_true']
phitemp2 = (nu2_dict['nu2_pt_true'] - nu2_dict['nu2_pt_regr'])/nu2_dict['nu2_pt_true']

etatemp1 = (nu1_dict['nu1_p_true'] - nu1_dict['nu1_p_regr'])/nu1_dict['nu1_p_true']
etatemp2 = (nu2_dict['nu2_p_true'] - nu2_dict['nu2_p_regr'])/nu2_dict['nu2_p_true']

fig, ax = plt.subplots(1,2,figsize=(12, 4))
ax[0].hist(phitemp1, bins=100, range=[-5,2], histtype="stepfilled", alpha=0.7, log=False, label= "nu1_ptres: true - pred")
ax[0].hist(phitemp2, bins=100, range=[-5,2], histtype="stepfilled", alpha=0.7, log=False, label= "nu2_ptres: true - pred")
ax[0].legend()

ax[1].hist(etatemp1, bins=100, range=[-5,2], histtype="stepfilled", alpha=0.7, log=False, label= "nu1_pres: true - pred")
ax[1].hist(etatemp2, bins=100, range=[-5,2], histtype="stepfilled", alpha=0.7, log=False, label= "nu2_pres: true - pred")
ax[1].legend()

plt.tight_layout()
#plt.savefig(os.path.join(tagdir,"tau_nu_pt.png"), dpi=300)
#plt.show()
#plt.xlabel("nu1 (true_phi - pred_phi) / true_phi")
#plt.show()
plt.savefig(os.path.join(tagdir,'nu_pt_p_res.png'), dpi=300)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#vis_tau_1_p4 = getp4(df_test_x, vars=["tau_1_pt", "tau_1_eta", "tau_1_phi", "tau_1_mass"], LVtype="ptetaphim")


vis_tau_1_p4 = vector.array({"pt": df_test_x["pt_tau_1"].to_numpy(),
                             "eta": df_test_x["eta_tau_1"].to_numpy(),
                             "phi": df_test_x["phi_tau_1"].to_numpy(),# + df2_met_raw_phi_test["met_phi_raw"].to_numpy(),
                             "M": df_test_x["mass_tau_1"].to_numpy()})
vis_tau_2_p4 = vector.array({"pt": df_test_x["pt_tau_2"].to_numpy(),
                             "eta": df_test_x["eta_tau_2"].to_numpy(),
                             "phi": df_test_x["phi_tau_2"].to_numpy(),# + df2_met_raw_phi_test["met_phi_raw"].to_numpy(),
                             "M": df_test_x["mass_tau_2"].to_numpy()})


# In[ ]:


vis_tau_2_p4


# In[ ]:


vis_Z_p4 = vis_tau_1_p4 + vis_tau_2_p4
m = vis_Z_p4.M


# In[ ]:


m


# In[ ]:


np.mean(m)


# In[ ]:





# In[ ]:


plt.hist(vis_Z_p4.M, range=[0,120], bins=50, histtype="stepfilled", alpha=0.7)
plt.show()


# In[ ]:


"""
true_full_tau_1_p4 = vis_tau_1_p4 + true_nu1_p4
true_full_tau_2_p4 = vis_tau_2_p4 + true_nu2_p4
true_Z_p4 = true_full_tau_1_p4 + true_full_tau_2_p4

regr_full_tau_1_p4 = vis_tau_1_p4 + regr_nu1_p4
regr_full_tau_2_p4 = vis_tau_2_p4 + regr_nu2_p4
regr_Z_p4 = regr_full_tau_1_p4 + regr_full_tau_2_p4

plt.hist(true_Z_p4.M, range=[0,120], bins=50, histtype="stepfilled", alpha=0.7)
plt.hist(regr_Z_p4.M, range=[0,120], bins=50, histtype="stepfilled", alpha=0.7)

plt.show()
"""


# In[ ]:





# In[ ]:


true_nu1_p4.pt


# In[ ]:


true_tau1p4 = vis_tau_1_p4 + true_nu1_p4
true_tau2p4 = vis_tau_2_p4 + true_nu2_p4
true_mass = (true_tau1p4 + true_tau2p4).M
true_mass, np.mean(true_mass)


# In[ ]:


regr_tau1p4 = vis_tau_1_p4 + regr_nu1_p4
regr_tau2p4 = vis_tau_2_p4 + regr_nu2_p4
regr_mass = (regr_tau1p4 + regr_tau2p4).M
regr_mass, np.mean(regr_mass)


# In[ ]:


vis_mass = (vis_tau_1_p4 + vis_tau_2_p4).mass


# In[ ]:


plt.figure(figsize=(8,6))

plt.hist(true_mass, range=[80,140], bins=30, histtype="stepfilled", alpha=0.7, label="visible tau + true nu")
plt.hist(regr_mass, range=[80,140], bins=30, histtype="stepfilled", alpha=0.7, label="visible tau + regressed nu")
plt.xlabel("$m_{tau-tau}$")
plt.legend()


plt.tight_layout()
plt.savefig(os.path.join(tagdir,'mass_tau_tau.png'), dpi=300)


# In[ ]:





# In[ ]:


target_regr, target_true


# In[ ]:


df.keys()


# In[ ]:


tau_vis_1 = getp4(df, vars=["pt_tau_1","eta_tau_1","phi_tau_1","mass_tau_1"], LVtype="ptetaphim")
tau_vis_2 = getp4(df, vars=["pt_tau_2","eta_tau_2","phi_tau_2","mass_tau_2"], LVtype="ptetaphim")
true_nu1_p4 = vector.array({"px": target_true[:,:1].reshape(-1),
                            "py": target_true[:,1:2].reshape(-1),
                            "pz": target_true[:,2:3].reshape(-1),
                            #"E": target_val_test[:,3:4]})
                            "E": np.sqrt(target_true[:,:1]**2 + target_true[:,1:2]**2 + target_true[:,2:3]**2).reshape(-1)})
true_nu2_p4 = vector.array({"px": target_true[:,3:4].reshape(-1),
                            "py": target_true[:,4:5].reshape(-1),
                            "pz": target_true[:,5:6].reshape(-1),
                            #"E": target_val_test[:,7:8]})
                            "E": np.sqrt(target_true[:,3:4]**2 + target_true[:,4:5]**2 + target_true[:,5:6]**2).reshape(-1)})
regr_nu1_p4 = vector.array({"px": target_regr[:,:1].reshape(-1),
                            "py": target_regr[:,1:2].reshape(-1),
                            "pz": target_regr[:,2:3].reshape(-1),
                            #"E": target_val_test[:,3:4]})
                            "E": np.sqrt(target_regr[:,:1]**2 + target_regr[:,1:2]**2 + target_regr[:,2:3]**2).reshape(-1)})
regr_nu2_p4 = vector.array({"px": target_regr[:,3:4].reshape(-1),
                            "py": target_regr[:,4:5].reshape(-1),
                            "pz": target_regr[:,5:6].reshape(-1),
                            #"E": target_val_test[:,7:8]})
                            "E": np.sqrt(target_regr[:,3:4]**2 + target_regr[:,4:5]**2 + target_regr[:,5:6]**2).reshape(-1)})


# In[ ]:


target_true[:,:1].reshape(-1)


# In[ ]:


tau1_full_p4 = tau_vis_1 + regr_nu1_p4
tau2_full_p4 = tau_vis_2 + regr_nu2_p4

pi1_p4 = getp4(df, vars=["pt_tau1pi_1","eta_tau1pi_1","phi_tau1pi_1","mass_tau1pi_1"], LVtype="ptetaphim")
pi2_p4 = getp4(df, vars=["pt_tau2pi_1","eta_tau2pi_1","phi_tau2pi_1","mass_tau2pi_1"], LVtype="ptetaphim")
pi01_p4 = getp4(df, vars=["pt_tau1pi0_1","eta_tau1pi0_1","phi_tau1pi0_1","mass_tau1pi0_1"], LVtype="ptetaphim")
pi02_p4 = getp4(df, vars=["pt_tau2pi0_1","eta_tau2pi0_1","phi_tau2pi0_1","mass_tau2pi0_1"], LVtype="ptetaphim")



# In[ ]:


tau1_full_p4.pt


# In[ ]:


import awkward as ak
from coffea.nanoevents.methods import vector
def getakp4(tau1_full_p4, pid):
    return ak.zip(
        {
            "pt": ak.Array(tau1_full_p4.pt[:,None]),
            "eta": ak.Array(tau1_full_p4.eta[:,None]),
            "phi": ak.Array(tau1_full_p4.phi[:,None]),
            "mass": ak.Array(tau1_full_p4.mass[:,None]),
            "pdgId": pid*ak.ones_like(ak.Array(tau1_full_p4.mass[:,None])),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )


# In[ ]:


tau1_full_p4 = getakp4(tau1_full_p4, 15)
tau2_full_p4 = getakp4(tau2_full_p4, 15)
pi1_p4 = getakp4(pi1_p4, 211)
pi2_p4 = getakp4(pi2_p4, 211)
pi01_p4 = getakp4(pi01_p4, 111)
pi02_p4 = getakp4(pi02_p4, 111)


# In[ ]:


tau1_full_p4.pt, pi02_p4.pt


# In[ ]:


tau1_decay = ak.concatenate([pi1_p4,pi01_p4], axis=1)
tau2_decay = ak.concatenate([pi2_p4,pi02_p4], axis=1)
tau1_decay.pt, tau2_decay.pt


# In[ ]:


from PhiCPComp import PhiCPComp
phicp_obj = PhiCPComp(cat="rhorho",
                      taum=tau1_full_p4,
                      taup=tau2_full_p4,
                      taum_decay=tau1_decay,
                      taup_decay=tau2_decay)
phi_cp, _ = phicp_obj.comp_phiCP()
phi_cp = ak.firsts(phi_cp, axis=1)
phi_cp


# In[ ]:


phi_cp


# In[ ]:


df["phicp"], df["phi_cp_det"]


# In[ ]:


phicp_det = phi_cp
phicp_gen = ak.Array(df["phicp"])
phicp_det_true_tau = ak.Array(df["phi_cp_det"])
phicp_det, phicp_gen, phicp_det_true_tau


# In[ ]:


#plt.figure(12,8)
plt.hist(ak.to_numpy(phicp_det), bins=25, label="Full Det", density=True, alpha=0.9, lw=2, histtype="step")
plt.hist(ak.to_numpy(phicp_det_true_tau), bins=25, label="Gen tau + det decays", density=True, alpha=0.9, lw=2, histtype="step")
plt.hist(ak.to_numpy(phicp_gen), bins=25, label="Full Gen", density=True, alpha=0.9, lw=2, histtype="step")


plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(tagdir,'PhiCP.png'), dpi=300)
#plt.show()
