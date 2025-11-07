import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import numpy as np
import json
import os 
import sys
from scipy import stats
from scipy import signal
from datetime import datetime 
import seaborn as sns
import joblib
from scipy.stats import pearsonr
import zipfile
from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    CNNEmbedding,
    PermutationInvariantEmbedding
)


#path_curr = os.path.dirname(os.path.abspath('.'))
sys.path.append(os.path.abspath(".."))
## Import TVB files:
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_dt01s import Parameter # type: ignore
parameters = Parameter()
from scipy.spatial.distance import mahalanobis
import tvb_model_reference.src.nuu_tools_simulation_human as tools # type: ignore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Ignore les avertissements de dépréciation pour éviter l'affichage continu
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torch.distributions import MultivariateNormal


################## PARAMÈTRES DE SIMULATION SBI ##################




# Paramètres de simulation par défaut
default_params = {
    'Qi': 5.0,                             # Adaptation globale
    'Iext': 0.01,                          # Courant externe
    'coupling_strength': 0.15,            # Couplage interrégional
    'ratio_coupling_EE_EI': 1.4,          # Ratio E→E / E→I
    'Qi_change_curr': 5.0,                # Valeur locale modifiée de Qi
    'node_name_Qichange': 'PCC',          # Nom des régions modifiées (pour nommer seulement, pas de reference de noms)
    'nodes_Qichange': [46, 47],           # Index des régions modifiées (noeuds associés à la région cible)
    'save_BOLD': True,                    # Sauvegarde du signal BOLD
    'add_transient': 0,                   # Transitoire initial (ms)
    'cut_time': 0,                        # Temps à couper (ms)
    'run_time': 100*10**3,                  # Durée totale (ms)
    'seed': 0                             # Graine aléatoire
}

# Observation cible (peut être modifiée indépendamment du modèle simulé)
params_obs = default_params.copy()
params_obs.update({
    'Qi': 5.2,                # Valeur modifiée uniquement pour l'observation
})

# Paramètres à inférer via SBI
list_inferer = ['Qi']        # Exemple : un seul paramètre local

# Bornes du prior
borne_min = torch.tensor([4.0])
borne_max = torch.tensor([6.0])

# Choix du modèle pour la posterior (density estimator)
MODEL_NN = 'maf'                         # Options : ['mdn', 'made', 'maf', 'maf_rqs', 'nsf']

# Paramètres d’inférence
NUM_SAMPLES = 100                         # Nb d’échantillons tirés pour estimer la postérieure
NUM_SIMULATION = 1000                     # Nb de simulations par round
NUM_ROUNDS = 5                           # Nb de rounds (utile pour multi-round inference)

# Informations sur l’expérience
EXP_NAME = 'Qi_1000_sim_real_obs_ctrl'  # Nom utilisé pour les dossiers et fichiers de sortie

# Observation réelle ou simulée
REAL_OBS = True                         # Si True : utilise une observation issue de données réelles

# Données patient spécifiques
CTRL = True                             # True pour sujet contrôle, False pour schizophrène
SUBJ_IDX = 0                             # Index du sujet dans les données
RANDOM_SC = False                        # True pour tirer un SC aléatoire
USE_PATIENT_SC = True                    # True pour utiliser le connectome réel du patient





################## FONCTIONS UTILISÉES ######################

# SIMULATION ET OBSERVATION FC 
def generate_BOLD_filename(params, folder_root='save_file/'):
    """
    Génère le chemin complet du fichier BOLD à partir des paramètres de simulation.

    Cette fonction utilise un dictionnaire de paramètres pour construire un nom de fichier unique,
    permettant de sauvegarder ou charger un fichier BOLD simulé, avec un nom explicite
    reflétant les conditions de simulation.

    Paramètres :
    - params : (dict) Dictionnaire contenant au minimum les clés suivantes :
        - 'Qi' : (float) Valeur du paramètre Qi.
        - 'node_name_Qichange' : (str) Nom du noeud auquel Qi est modifié.
        - 'Qi_change_curr' : (float) Valeur appliquée de Qi dans le noeud cible.
        - 'ratio_coupling_EE_EI' : (float) Ratio entre le couplage E→E et E→I.
        - 'coupling_strength' : (float) Intensité du couplage.
        - 'seed' : (int) Graine aléatoire pour reproductibilité.
        - 'Iext' : (float) Intensité du courant externe.
    - folder_root : (str, optionnel) Dossier racine pour la sauvegarde (par défaut : 'save_file/').

    Retour :
    - (str) Chemin complet du fichier BOLD, formaté de manière cohérente.
      Exemple : 'save_file/sig_BOLD_b_0_Qi_5.0PCC5EtoEIratio1.4_coupling0.15seed0_noise0.01.npy'
    """
    Qi = params['Qi']
    node_name_Qichange = params['node_name_Qichange']
    Qi_change_curr = params['Qi_change_curr']
    ratio_coupling_EE_EI = params['ratio_coupling_EE_EI']
    coupling_strength = params['coupling_strength']
    seed = params['seed']
    Iext = params['Iext']
    return f'{folder_root}sig_BOLD_b_0_Qi_{str(Qi)}{node_name_Qichange}{Qi_change_curr}EtoEIratio{ratio_coupling_EE_EI}_coupling{coupling_strength}seed{seed}_noise{Iext}.npy'

def simulator_BOLD(params = default_params) :
    """
    Exécute une simulation de l'activité neuronale dans un réseau cérébral, puis génère et retourne les signaux BOLD correspondants.

    Cette fonction utilise les paramètres fournis dans un dictionnaire `params` pour configurer et exécuter une simulation
    du modèle dynamique de connectome. Elle simule l'activité spontanée ou stimulée d’un réseau de régions cérébrales,
    applique une convolution hémodynamique (HRF) pour générer des signaux BOLD, et retourne les signaux sous-échantillonnés.
    Si `save_BOLD` est activé, les résultats sont enregistrés sur disque.

    Paramètres :
    - params : (dict) Dictionnaire contenant les clés suivantes :
        - 'Qi' : (float) Valeur globale du paramètre d'adaptation (Q_i).
        - 'Qi_change_curr' : (float) Nouvelle valeur de Q_i appliquée à certains noeuds cibles.
        - 'node_name_Qichange' : (str) Nom descriptif du/des noeud(s) modifié(s).
        - 'nodes_Qichange' : (list of int) Indices des noeuds où Q_i est modifié.
        - 'ratio_coupling_EE_EI' : (float) Ratio des couplages inter-régionaux E→E / E→I.
        - 'coupling_strength' : (float) Force globale de couplage du connectome.
        - 'Iext' : (float) Intensité du courant d'entrée externe (bruit ou input constant).
        - 'seed' : (int) Graine aléatoire utilisée pour les conditions initiales et le bruit.
        - 'save_BOLD' : (bool) Si True, enregistre les signaux BOLD sous forme de fichier `.npy`.
        - 'add_transient' : (int) Durée additionnelle (ms) du segment initial ignoré.
        - 'cut_time' : (int) Temps initial à ignorer avant analyse (en ms).
        - 'run_time' : (int) Durée effective de la simulation (en ms).

    Retour :
    - BOLD_subsamp : (np.ndarray) Matrice des signaux BOLD sous-échantillonnés pour chaque région cérébrale.
                     Dimensions : [n_regions, n_timepoints]

    Notes :
    - Le nom du fichier BOLD de sortie encode tous les paramètres principaux pour reproductibilité.
    - Le signal neuronal est obtenu via des taux de décharge (E/I), puis transformé via convolution HRF.
    - Un réseau par défaut (connectome) est utilisé, et certaines régions peuvent être stimulées (ex. insula).
    - Le système utilise Torch, NumPy et Scipy, et dépend d'un ensemble de paramètres définis dans `parameters`.

    Exceptions :
    - Si les noeuds spécifiés sont invalides ou si un fichier requis n'est pas disponible, des erreurs peuvent apparaître.
    """


    Qi = params['Qi']
    node_name_Qichange = params['node_name_Qichange']
    Qi_change_curr = params['Qi_change_curr']
    ratio_coupling_EE_EI = params['ratio_coupling_EE_EI']
    coupling_strength = params['coupling_strength']
    seed = params['seed']
    Iext = params['Iext']
    Qi_change_curr = params['Qi_change_curr']
    nodes_Qichange = params['nodes_Qichange']
    save_BOLD = params['save_BOLD']
    add_transient = params['add_transient']
    cut_time = params['cut_time']
    run_time = params['run_time']
    
    ## Set the parameters of the simulation

    bvals = 0 # List of values of adaptation strength which will vary the brain state
    
    
    ## Set the parameters of the stimulus (choose stimval = 0 to simulate spontaneous activity)

    stimval = 0. # Hz, stimulus strength
    stimdur = 50 # ms, duration of the s



    ## Define a location to save the files; create the folder if necessary:

    folder_root = 'save_file/' # /!\ TO CHANGE TO ADAPT ON YOUR PLATFORM /!\
    try:
        os.listdir(folder_root)
    except:
        os.mkdir(folder_root)

    # set the seeds
    

    # Nodes that can be used
    """PCC_nodes = [46,47]
    Pcun_nodes = [50,51]
    Ang_nodes = [14,15]
    Insula_nodes = [18,19] # agranular insula
    """
    stim_region = np.array([18,19]) # left and right insula


    # utiliser les SC des données réelles
    if USE_PATIENT_SC:
        sc_paths = extract_SC_patient()
        # injection dans TVB
        parameters.parameter_connection_between_region["path_weight"]     = sc_paths["weights"]
        parameters.parameter_connection_between_region["path_length"]     = sc_paths["tract_lengths"]
        parameters.parameter_connection_between_region["path_region_labels"] = sc_paths["region_labels"]
        parameters.parameter_connection_between_region["path_centres"]    = sc_paths["centres"]


    ## Run simulations (takes time!)

    
    ## Set up the simulator:
    simulator = tools.init(parameters.parameter_simulation,
                                parameters.parameter_model,
                                parameters.parameter_connection_between_region,
                                parameters.parameter_coupling,
                                parameters.parameter_integrator,
                                parameters.parameter_monitor,
                                my_seed = int(seed)) # seed setting for noise generator, has to be int for json
    # Set parameters
    # adap
    parameters.parameter_model['b_e'] = bvals
        
    # local I conductance
    if simulator.number_of_nodes > 1:
        if isinstance(Qi, torch.Tensor):
            # Si Qi est un Tensor, on le convertit en numpy.ndarray avant de le multiplier
            Qi_allreg = np.ones(simulator.number_of_nodes) * Qi.numpy()
        else:
            # Si Qi n'est pas un Tensor, on peut simplement multiplier directement avec la valeur Qi
            Qi_allreg = np.ones(simulator.number_of_nodes) * Qi
        #print('Qi_allreg = ', Qi_allreg.shape)
        Qi_allreg = Qi_allreg.flatten()
        Qi_allreg[nodes_Qichange] = Qi_change_curr  # Modifie les indices spécifiques
    else:
        raise ValueError(f"Le nombre de noeuds dans le simulateur est trop petit ({simulator.number_of_nodes}), "
                        f"vérifie la configuration de ton réseau et les indices des noeuds.")

    parameters.parameter_model['Q_i'] = list(Qi_allreg)
        
    # E to I inter-regional coupling strength
    K_e = parameters.parameter_model['K_ext_e']
    K_i = 0
    type_K_e = type(K_e)
    parameters.parameter_model['ratio_EI_EE'] = ratio_coupling_EE_EI
    parameters.parameter_model['K_ext_i'] = type_K_e(K_i)
    #print('ratio EtoI/EtoE = ', parameters.parameter_model['ratio_EI_EE'])

    parameters.parameter_model['external_input_ex_ex']=Iext
    parameters.parameter_model['external_input_in_ex']=Iext
    
    weight = list(np.zeros(simulator.number_of_nodes))
    for reg in stim_region:
        weight[reg] = stimval # region and stimulation strength of the region 0 
            
        # random transient length for phase randomisation
        


    
    cut_transient = cut_time + add_transient# ms, length of the discarded initial segment # 4000 to be similar to the real dataset
    run_sim =  run_time  + cut_transient# for feature attribution testing to match human fMI data length 100 s # temps de simulation en ms
            
    parameters.parameter_stimulus["tau"]= stimdur # stimulus duration [ms]
    parameters.parameter_stimulus["T"]= 2000.0 # interstimulus interval [ms]
    parameters.parameter_stimulus["weights"]= weight
    parameters.parameter_stimulus["variables"]=[0] #variable to kick
        
        
    # parameters.parameter_connection_between_region["path"]=connectome_path
    # attempt use default in param file?
    parameters.parameter_coupling["parameter"]["a"]=coupling_strength
    
    
    #print('b_e =', bvals,', Iext = ', Iext,', Qi = ', Qi, 'changed in ', simulator.connectivity.region_labels[nodes_Qichange[0]])
    
    # random initial conditions depending on seed
    np.random.seed(seed)
        
    ratemax_init = 0.001 # max initial rate, in kHz
    adapmax_init = 20.*bvals # max initial E adaptation, pA
    initE = np.random.rand()*ratemax_init
    initI = initE*4  #E/I balance
    initW = initE**2*adapmax_init/ratemax_init**2 # avoid explosion when large rate 
    # randomise initial rate and adap for phase randomisation 
        
    parameters.parameter_model['initial_condition']['E'] = [initE, initE]
    parameters.parameter_model['initial_condition']['I'] = [initI, initI]
    parameters.parameter_model['initial_condition']['W_e'] = [initW, initW]

    # one stim
    parameters.parameter_stimulus['onset'] = cut_transient + 0.5*(run_sim-cut_transient)
    
    # repeated stim
    parameters.parameter_stimulus['onset'] = cut_transient + parameters.parameter_stimulus["T"]/2
    stim_time = parameters.parameter_stimulus['onset']
        
    stim_region_name_l = simulator.connectivity.region_labels[stim_region]
    '''# save a file with the name of the stimulated region
    np.save(folder_root + 'stim_region_labels.npy', stim_region_name_l)
    '''

    if len(stim_region_name_l) > 1:
        name_large_regions = []
        for reg_name_curr in stim_region_name_l:
            idx_char = 0
            while reg_name_curr[idx_char].islower() and \
                idx_char < len(reg_name_curr):
                    idx_char += 1
            name_large_regions.append(reg_name_curr[:idx_char])
        stim_region_name = '-'.join(np.unique(name_large_regions))
    else:
        stim_region_name = ''.join(stim_region_name_l)
    #print(stim_region_name)
                    
    # simulation saving path
    # Attention!!!!!! : Qi = Qi global, Qi_change = Qi in the region
    path_save = folder_root  +'_Qi_'+str(Qi)+node_name_Qichange+str(Qi_change_curr)+\
        'EtoEIratio'+str(ratio_coupling_EE_EI)+'_coupling'+str(coupling_strength)+\
        'seed'+str(seed)+'_noise'+str(Iext) +'/'
    #print(path_save)
    parameters.parameter_simulation['path_result'] = path_save

    # read the parameters file and check the path
        
    if not os.path.exists(path_save): # new folder if not yet existing
        os.makedirs(path_save)
    np.save(path_save + 'cut_transient_seed.npy', np.array(cut_transient))
        
    # running simulations
    simulator = tools.init(parameters.parameter_simulation,
                                parameters.parameter_model,
                                parameters.parameter_connection_between_region,
                                parameters.parameter_coupling,
                                parameters.parameter_integrator,
                                parameters.parameter_monitor,
                                parameter_stimulation=parameters.parameter_stimulus,
                                my_seed = int(seed))
    if stimval != False:
        print ('    Stimulating for {1} ms, {2} nS in the {0}\n'.format(simulator.connectivity.region_labels[stim_region],parameters.parameter_stimulus['tau'],stimval))
    #print(run_sim)
    tools.run_simulation(simulator,
                        run_sim,                            
                        parameters.parameter_simulation,
                        parameters.parameter_monitor)

    try:
        run_sim_transient = run_sim # ms, total simulation time 
        tinterstim = parameters.parameter_stimulus["T"]
        time_after_last_stim = (run_sim_transient - cut_transient)//tinterstim*tinterstim + cut_transient
    
        time_begin_all = np.arange(cut_transient, time_after_last_stim, tinterstim)
        Esig_alltime = []
        Isig_alltime = []
            
        for time_begin in time_begin_all: 
            try:# to avoid boundary error...
                    #print('loading from', time_begin, 'to', time_begin + tinterstim, ' ms')
                result = tools.get_result(path_save,time_begin,time_begin + tinterstim)
    
                '''fill variables'''
                if len(result) > 0:
                        
                    time_s = result[0][0]*1e-3 - result[0][0][0]*1e-3 #from ms to sec
                    Esig_alltime.append(result[0][1][:,0,:]*1e3)
                    Isig_alltime.append(result[0][1][:,1,:]*1e3)
                        
            except:
                print(time_begin, ' not found')
            
        Esig = np.concatenate(np.array(Esig_alltime), axis = 0)
            
        Isig = np.concatenate(np.array(Isig_alltime), axis = 0)
            
        EIsig = 0.8*Esig + 0.2*Isig 
            
        # BOLD
        if save_BOLD:
            #print('bold...')
            dt_BOLD = 1. # s
            dt = time_s[1] - time_s[0]
            #print('dt = ',str(np.round(dt,3)))
                
            ratio_dt = int(dt_BOLD/dt)
            kernel_hrf = np.load('kernel_hrf_dt' + str(np.round(dt,
                                                            3)) + '.npy')
            FR_sum = result[0][1][:,0,:]*1e3*0.8 \
                    + result[0][1][:,1,:]*1e3*0.2
            BOLD_curr = []
            for idx_reg in range(len(FR_sum[0])):
                conv = signal.fftconvolve(EIsig[:,idx_reg],
                                                kernel_hrf, mode = 'valid')
                BOLD_curr.append(conv)
                    
            # subsample
            BOLD_subsamp = []
            for BOLD_reg in np.array(BOLD_curr): # loop over regions
                conv_subsamp = np.mean(np.reshape(BOLD_reg[:len(BOLD_reg)//\
                                                                ratio_dt*ratio_dt], 
                            (int(len(BOLD_reg)/ratio_dt), ratio_dt)), axis = 1)
                BOLD_subsamp.append(conv_subsamp)
            BOLD_subsamp = np.array(BOLD_subsamp)
                
            # save file
             
            filename = generate_BOLD_filename(params)
            np.save(filename, BOLD_subsamp)
            return BOLD_subsamp
    except:
        print('pass seed ', path_save, ' not found')

def simulator_FC(theta, params = default_params, list_inferer = list_inferer):
    """
    Simule une matrice de connectivité fonctionnelle (FC) à partir de signaux BOLD,
    en utilisant un ensemble de paramètres spécifiés pour l'inférence.

    La fonction prend en entrée un vecteur de paramètres `theta`, qui vient écraser 
    certaines valeurs dans le dictionnaire `params` selon la liste `list_inferer`.
    Elle vérifie ensuite si les signaux BOLD simulés correspondants existent déjà 
    (via un nom de fichier généré automatiquement). Si oui, elle les charge ; sinon, 
    elle appelle la fonction `simulator_BOLD` pour les générer. La matrice FC est 
    ensuite calculée via la corrélation des signaux BOLD, puis vectorisée.

    Paramètres
    ----------
    theta : torch.Tensor
        Tensor de forme [batch_size, n_params] contenant les valeurs des paramètres 
        à inférer (ex. Qi, Iext, coupling_strength, etc.).

    params : dict, optionnel
        Dictionnaire contenant les valeurs par défaut de tous les paramètres du modèle. 
        Ceux listés dans `list_inferer` seront remplacés par les valeurs de `theta`.

    list_inferer : list of str
        Liste des clés dans `params` à remplacer par les valeurs contenues dans `theta`.

    Retour
    ------
    torch.Tensor
        Tensor de forme [1, n_edges] représentant la partie supérieure aplatie de la 
        matrice de connectivité fonctionnelle (hors diagonale).
    """
    
    # # Arrondir les paramètres pour éviter des doublons presque identiques
    # params = [Qi, Qi_change_curr, ratio_coupling_EE_EI, coupling_strength, Iext]
    # decimals = [3, 2, 1, 2, 4]

    # # Arrondi
    # params = [round_param_general(p, d) for p, d in zip(params, decimals)]

    # # Réaffectation
    # Qi, Qi_change_curr, ratio_coupling_EE_EI, coupling_strength, Iext = params
    
    
    params_c = params.copy()

    # Gestion des deux cas : theta dict (observation) ou tensor (batch simulations)
    if isinstance(theta, dict):
        for key in list_inferer:
            params_c[key] = float(theta[key])  # valeur unique
    else:
        for i, key in enumerate(list_inferer):
            # Assure-toi que theta est 1D ici (un seul échantillon)
            value = theta[i].item() if isinstance(theta[i], torch.Tensor) else float(theta[i])
            params_c[key] = value




    # Génération du nom de fichier basé sur les paramètres arrondis
    filename = generate_BOLD_filename(params_c)
    #print('filename:', filename)
    # Vérifier si le fichier BOLD existe
    if os.path.exists(filename):
        bold = np.load(filename, allow_pickle=True)
        #print('find it!')
    else:
        #print('running simulation...')
        bold = simulator_BOLD(params_c)


     # Calcul de la matrice FC
    FC = np.corrcoef(bold)

    # Extraction de la partie triangulaire supérieure (hors diagonale)
    triu_indices = np.triu_indices_from(FC, k=1)
    FC_vector = FC[triu_indices]

    return torch.tensor(FC_vector, dtype=torch.float32)

def generate_observation(
    params=params_obs,
    list_inferer=list_inferer,
    real_obs=False,
    ctrl=True,
    subject_idx=SUBJ_IDX,
    data_path="/volatile/home/lb283126/Documents/Code/tvb_model_reference/data/real_data/FC_data_flattened.npz"
):
    """
    Génère une observation pour l'inférence bayésienne.

    Deux modes sont disponibles :
    - Observation simulée : utilise un modèle neuronal pour générer une FC à partir de paramètres.
    - Observation réelle : charge une matrice FC réelle (sujets sains ou schizophrènes).

    Paramètres
    ----------
    params : dict
        Dictionnaire des paramètres par défaut du modèle.

    list_inferer : list of str
        Liste des noms de paramètres à inférer (clés extraites de `params`).

    real_obs : bool
        Si True, charge une observation réelle depuis un fichier .npz. Sinon, simule une FC.

    ctrl : bool
        Spécifie si l'observation réelle doit venir du groupe contrôle (True) ou schizophrène (False).
        Ignoré si `real_obs=False`.

    subject_idx : int
        Index du sujet à utiliser dans les données réelles (par défaut : 0).

    data_path : str
        Chemin vers le fichier .npz contenant les données réelles.

    Retour
    ------
    - Si `real_obs` est False :
        tuple (theta: dict, FC: torch.Tensor)
        theta : dictionnaire des paramètres à inférer.
        FC : matrice FC simulée, vectorisée.

    - Si `real_obs` est True :
        FC : torch.Tensor
        Observation réelle, matrice FC vectorisée.
    """
    if not real_obs:
        theta = {}
        for p in list_inferer:
            if p not in params:
                raise ValueError(f"Paramètre '{p}' absent de `params`.")
            theta[p] = params[p]
        return theta, simulator_FC(theta, params)

    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Fichier de données introuvable : {data_path}")

        data = np.load(data_path, allow_pickle=True)

        if ctrl:
            key = "X_ctrl"
        else:
            key = "X_schz"

        if key not in data:
            raise KeyError(f"Clé '{key}' absente dans le fichier de données.")

        FC_matrix = data[key][subject_idx,:]
        FC_tensor = torch.tensor(FC_matrix, dtype=torch.float32)

        return None, FC_tensor

def extract_SC_patient(SUBJ_IDX=None, CTRL=True, RANDOM_SC=False, base_dir="data/real_data/"):
    """
    Sélectionne et extrait la connectivité structurelle d’un patient depuis un .zip TVB-compatible.

    Paramètres
    ----------
    SUBJ_IDX : int or None
        Index du sujet (entre 0 et 26), sinon choisi aléatoirement si RANDOM_SC est True.

    CTRL : bool
        Si True, sélectionne un patient contrôle. Sinon, un patient schizophrène.

    RANDOM_SC : bool
        Si True, un patient aléatoire est tiré, dans le groupe CTRL ou SCHZ selon `CTRL`.

    base_dir : str
        Dossier contenant les fichiers zip générés avec generate_all_SC_tvb_zips().

    Retourne
    -------
    dict
        Dictionnaire contenant les chemins vers :
        - 'weights'
        - 'tract_lengths'
        - 'region_labels'
        - 'centres'
    """
    os.makedirs("tmp_connectome", exist_ok=True)

    if RANDOM_SC or SUBJ_IDX is None:
        SUBJ_IDX = np.random.randint(0, 26)

    group = "ctrl" if CTRL else "schz"
    zip_name = f"patient_{group}_{SUBJ_IDX}.zip"
    zip_path = os.path.join(base_dir, zip_name)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[✗] Fichier {zip_path} introuvable.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tmp_connectome")

    return {
        "weights": "tmp_connectome/weights.npy",
        "tract_lengths": "tmp_connectome/tract_lengths.npy",
        "region_labels": "tmp_connectome/region_labels.txt",
        "centres": "tmp_connectome/centres.txt"
    }


# Evaluation et visualisation des prédictions



def compute_predictions_from_posterior(posterior,obs,simulator,list_inferer,N_samples=10,params=default_params):
    """
    Génère des prédictions à partir des échantillons tirés du posterior conditionné sur une observation.

    Cette fonction génère des prédictions en simulant l'activité neuronale à partir des échantillons
    du posterior pour les paramètres à inférer. Les autres paramètres sont fixés à leurs valeurs par défaut,
    spécifiées dans `params`.

    Paramètres :
    -----------
    posterior : Posterior
        Distribution posterior estimée par le modèle SBI.
    obs : torch.Tensor
        Observation pour conditionner l'échantillonnage du posterior.
    simulator : function
        Fonction prenant en entrée un ensemble de paramètres theta et retournant des données simulées (BOLD, FC, etc.).
    list_inferer : list of str
        Liste des noms des paramètres à inférer.
    N_samples : int, optional
        Nombre d’échantillons à tirer du posterior (par défaut 10).
    params : dict, optional
        Dictionnaire contenant les valeurs par défaut de tous les paramètres du modèle.

    Retour :
    -------
    theta_posterior : torch.Tensor
        Échantillons tirés du posterior (shape [N_samples, num_dim]).
    x_predictive_all : torch.Tensor
        Prédictions simulées par le modèle (shape [N_samples, dim]).
    """

    # 1. Échantillonnage
    theta_posterior = posterior.sample((N_samples,), x=obs)

    # 2. Simulation parallèle
    with joblib.Parallel(n_jobs=-1) as parallel:
        x_predictive_all = parallel(
            joblib.delayed(simulator)(
                theta=theta_posterior[i],
                params=params,
                list_inferer=list_inferer
            )
            for i in range(N_samples)
        )

    # 3. Mise en forme des résultats
    x_predictive_all = np.vstack([
        x.detach().cpu().numpy() if hasattr(x, "detach") else x
        for x in x_predictive_all
    ])

    return theta_posterior.numpy(), x_predictive_all

def compute_pearson_scores(predictions, observation):
    """
    Calcule les scores d’erreur entre des prédictions et une observation cible
    en utilisant la distance de Pearson.

    Paramètres :
    - predictions : np.ndarray de forme (N, D), où N est le nombre d’échantillons
                    et D la dimension de chaque échantillon.
    - observation : np.ndarray de forme (D,), la cible à comparer.

    Retourne :
    - un dictionnaire avec :
        - la distance de Pearson moyenne
        - l’écart-type
        - la meilleure (minimale) distance
        - la pire (maximale) distance
    """

    observation = observation.squeeze()  # (1, 2278) → (2278,)


    distances = []
    for pred in predictions:
        corr, _ = pearsonr(pred, observation)
        distances.append(1 - corr)

    distances = np.array(distances)

     
    stats = {
    'mean_pearson_distance': np.mean(distances),
    'std_pearson_distance': np.std(distances),
    'min_pearson_distance': np.min(distances),
    'max_pearson_distance': np.max(distances)}

    print(stats)

def visualize_predictions_with_pca(
    x_predictive_all,
    obs,
    pca_components=3,
    figsize=(6, 6),
    labels_prefix="$x$"):
    """
    Visualise les prédictions en réduisant leur dimension avec PCA.

    Paramètres :
    - x_predictive_all : np.ndarray, shape (N_samples, dim)
        Prédictions issues du simulateur.
    - obs : np.ndarray, shape (dim,)
        Observation cible à comparer.
    - pca_components : int
        Nombre de composantes principales à garder pour la PCA.
    - figsize : tuple
        Taille de la figure pour le pairplot.
    - labels_prefix : str
        Préfixe pour les labels des axes.

    Affiche :
    - Un pairplot des données projetées via PCA.
    """

    # 1. PCA
    pca = PCA(n_components=pca_components)
    x_projected = pca.fit_transform(x_predictive_all)
    obs_projected = pca.transform(obs.reshape(1, -1))

    # 2. Labels
    labels = [f"{labels_prefix}_{i+1}" for i in range(pca_components)]

    # 3. Affichage
    fig, ax = analysis.pairplot(
        x_projected,
        points=obs_projected,
        figsize=figsize,
        labels=labels
    )

    plt.sca(ax[min(1, pca_components - 1)])  # Évite l'erreur d'index
    plt.show()

def mse_theta(true_theta, posterior_samples):
    """
    Calcule le MSE entre le vrai theta et les échantillons du posterior.
    - true_theta : (d,) array
    - posterior_samples : (N, d) array
    """
    print("MSE theta :", np.mean((posterior_samples - true_theta) ** 2))

def mahalanobis_distance(true_theta, posterior_samples):
    """
    Calcule la distance de Mahalanobis entre theta vrai et les échantillons.
    """
    cov = np.cov(posterior_samples.T)
    cov_inv = np.linalg.pinv(cov)  # au cas où mal conditionné
    mean_post = np.mean(posterior_samples, axis=0)
    return mahalanobis(true_theta, mean_post, cov_inv)

def check_credible_interval(true_theta, posterior_samples, alpha=0.9):
    """
    Vérifie si chaque composante de theta_true est dans l’intervalle crédible.
    Retourne un tableau booléen de shape (d,).
    """
    lower = np.percentile(posterior_samples, (1 - alpha) / 2 * 100, axis=0)
    upper = np.percentile(posterior_samples, (1 + alpha) / 2 * 100, axis=0)
    return (true_theta >= lower) & (true_theta <= upper)

def posterior_calibration(true_theta_list, posterior_samples_list, alpha_list=[0.5, 0.9, 0.95, 0.99]):
    """
    Évalue la calibration d’un ensemble de postériori en vérifiant si les vrais paramètres
    se trouvent dans les intervalles crédibles des distributions postérieures.

    Paramètres :
    ----------
    true_theta_list : list of np.ndarray
        Liste des vrais paramètres (true theta) utilisés pour générer les observations.
        Chaque élément est un array de dimension (D,) où D est le nombre de paramètres.

    posterior_samples_list : list of np.ndarray
        Liste des échantillons postérieurs associés à chaque observation.
        Chaque élément est un array de dimension (N_samples, D).

    alpha_list : list of float
        Liste des niveaux d’intervalles crédibles à évaluer (e.g. 0.5 pour 50%).

    Retour :
    -------
    coverage_dict : dict
        Dictionnaire contenant le taux de couverture réel pour chaque niveau alpha.
        Par exemple, {0.5: 0.48, 0.9: 0.88} indique que l'intervalle à 50% couvre 48% des vrais paramètres.
    """
    assert len(true_theta_list) == len(posterior_samples_list), "Listes de true_theta et posterior_samples doivent avoir la même taille"

    n_tests = len(true_theta_list)
    dim = true_theta_list[0].shape[0]
    coverage_dict = {}

    for alpha in alpha_list:
        count_in_credible = 0

        for true_theta, posterior_samples in zip(true_theta_list, posterior_samples_list):
            # Calcul des bornes de l’intervalle crédible par dimension
            lower = np.percentile(posterior_samples, 100 * (1 - alpha) / 2, axis=0)
            upper = np.percentile(posterior_samples, 100 * (1 + alpha) / 2, axis=0)

            # Vérifie si chaque valeur vraie est bien dans son intervalle (pour chaque dimension)
            is_in_interval = np.logical_and(true_theta >= lower, true_theta <= upper)
            if np.all(is_in_interval):  # Tu peux aussi mettre np.any(is_in_interval) selon ce que tu veux tester
                count_in_credible += 1

        # Proportion de cas où tous les paramètres sont bien couverts par l'intervalle crédible
        coverage_dict[alpha] = count_in_credible / n_tests

    return coverage_dict

def quick_stats(x, name="x"):
    """
    Affiche un résumé statistique simple mais informatif d'un tenseur 2D (n_ex, n_features).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    mean_per_feature = np.mean(x, axis=0)
    std_per_feature = np.std(x, axis=0)

    print(f"--- Stats sur {name} ---")
    print(f"Shape : {x.shape}")
    print(f"Moyenne globale : {np.mean(mean_per_feature):.4f}")
    print(f"Écart-type global : {np.mean(std_per_feature):.4f}")
    print(f"Min absolu : {x.min():.4f}")
    print(f"Max absolu : {x.max():.4f}")
    print(f"Std min/max parmi features : {std_per_feature.min():.4f} / {std_per_feature.max():.4f}")
    print()


def vector_to_symmetric_matrix(vec, size):
    """Reconstruit une matrice symétrique à partir de la partie triangulaire supérieure sans diagonale."""
    mat = np.zeros((size, size))
    triu_indices = np.triu_indices(size, k=1)
    mat[triu_indices] = vec
    mat = mat + mat.T
    return mat


def log_experiment_results(
    true_theta,
    posterior_samples,
    predictions,
    observation,
    borne_min,
    borne_max,
    posterior_nn_model,
    experiment_name='simulation_based_inference',
    list_inferer=None,
    time_elapsed=0,
    num_samples=0,
    num_simulations=0,
    output_dir="results",
    file_name="experiences.json"
):
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from scipy.stats import pearsonr

    if isinstance(posterior_samples, torch.Tensor):
        posterior_samples = posterior_samples.detach().cpu().numpy()
    if isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()
    if list_inferer is None:
        list_inferer = [f"param_{i}" for i in range(posterior_samples.shape[1])]

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures", experiment_name)
    os.makedirs(fig_dir, exist_ok=True)

    # Calculs sur la postérieure
    posterior_mean = np.mean(posterior_samples, axis=0)
    posterior_std = np.std(posterior_samples, axis=0)
    lower = np.percentile(posterior_samples, 5, axis=0)
    upper = np.percentile(posterior_samples, 95, axis=0)

    # Résultats de base
    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "time_elapsed_sec": time_elapsed,
        "borne_min": borne_min.tolist(),
        "borne_max": borne_max.tolist(),
        "posterior_nn_model": str(posterior_nn_model),
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "parameters": list_inferer,
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": posterior_std.tolist(),
        "credible_interval_90": {
            "lower": lower.tolist(),
            "upper": upper.tolist()
        },
        "fc_saved": True
    }

    # Si true_theta existe → ajouter les performances
    if true_theta is not None:
        if isinstance(true_theta, torch.Tensor):
            true_theta = true_theta.detach().cpu().numpy()
        mse = np.mean((posterior_samples - true_theta) ** 2, axis=0)
        mse_std = np.std((posterior_samples - true_theta) ** 2, axis=0)
        mae = np.mean(np.abs(posterior_samples - true_theta), axis=0)
        mae_std = np.std(np.abs(posterior_samples - true_theta), axis=0)
        inside_interval = ((true_theta >= lower) & (true_theta <= upper)).tolist()

        results.update({
            "true_theta": true_theta.tolist(),
            "mse": mse.tolist(),
            "mse_std": mse_std.tolist(),
            "mae_mean": mae.tolist(),
            "mae_std": mae_std.tolist(),
            "credible_interval_90": {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "inside": inside_interval,
            }
        })

    # Distance de Pearson
    pearson_distances = [1 - pearsonr(pred, observation.squeeze())[0] for pred in predictions]
    best_idx = int(np.argmin(pearson_distances))
    pearson_stats = {
        "mean_pearson_distance": float(np.mean(pearson_distances)),
        "std_pearson_distance": float(np.std(pearson_distances)),
        "min_pearson_distance": float(np.min(pearson_distances)),
        "max_pearson_distance": float(np.max(pearson_distances)),
    }
    results["pearson_stats"] = pearson_stats
    results["best_sample_index"] = best_idx

    # Enregistrement du fichier JSON
    path = os.path.join(output_dir, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(results)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=4)

    # ---------- VISUALISATIONS ----------

    for i, name in enumerate(list_inferer):
        plt.figure(figsize=(6, 4))
        plt.hist(posterior_samples[:, i], bins=30, color="skyblue", alpha=0.7, edgecolor='black', label="Posterior")
        if true_theta is not None:
            plt.axvline(true_theta[i], color="red", linestyle="--", label="True value")
        plt.axvline(lower[i], color="black", linestyle="--", alpha=0.5, label="5% CI")
        plt.axvline(upper[i], color="black", linestyle="--", alpha=0.5, label="95% CI")
        plt.title(f"Posterior distribution of {name}")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"posterior_{name}.png"))
        plt.close()

    # Boxplot
    plt.figure()
    sns.boxplot(data=posterior_samples)
    plt.xticks(ticks=range(len(list_inferer)), labels=list_inferer)
    plt.title("Boxplot of posterior samples")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "posterior_boxplot.png"))
    plt.close()

    # Matrice de corrélation
    if posterior_samples.shape[1] > 1:
        corr = np.corrcoef(posterior_samples.T)
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, xticklabels=list_inferer, yticklabels=list_inferer, cmap="coolwarm", center=0)
        plt.title("Posterior correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "posterior_correlation_matrix.png"))
        plt.close()

    # Scatter si 2D
    if posterior_samples.shape[1] == 2:
        plt.figure()
        plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.5, edgecolor="k")
        if true_theta is not None:
            plt.scatter(true_theta[0], true_theta[1], color="red", label="True value")
        plt.xlabel(list_inferer[0])
        plt.ylabel(list_inferer[1])
        plt.title("Scatter plot of posterior samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "posterior_scatter.png"))
        plt.close()

    # FC comparison
    best_fc = predictions[best_idx]
    fc_dim = int((1 + np.sqrt(1 + 8 * len(best_fc))) // 2)
    FC_pred = vector_to_symmetric_matrix(best_fc, fc_dim)
    FC_obs = vector_to_symmetric_matrix(observation, fc_dim)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(FC_obs, cmap="viridis", cbar=True)
    plt.title("FC Observation")

    plt.subplot(1, 2, 2)
    sns.heatmap(FC_pred, cmap="viridis", cbar=True)
    plt.title("Best FC Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "best_FC_matrix_comparison.png"))
    plt.close()

    # Sauvegarde FC matrices
    np.save(os.path.join(fig_dir, "best_predicted_fc.npy"), best_fc)
    np.save(os.path.join(fig_dir, "fc_observation.npy"), observation)

    print(f"[✓] Résultats et figures enregistrés dans : {fig_dir}")


# INFERENCE

def single_round_inference(
    simulator_func = simulator_FC,
    params_obs = params_obs,
    list_inferer = list_inferer,
    borne_min = borne_min,
    borne_max = borne_max,
    MODEL_NN=MODEL_NN,
    EXP_NAME=EXP_NAME,
    NUM_SIMULATION=NUM_SIMULATION,
    NUM_SAMPLES=NUM_SAMPLES,
    REAL_OBS=REAL_OBS,
    save_dir="results"
):
    """
    Réalise une inférence SBI en un seul round avec simulation et estimation postérieure.

    Paramètres
    ----------
    simulator_func : callable
        Fonction de simulation (ex : simulator_FC).
    params_obs : dict
        Paramètres par défaut des observations simulées.
    list_inferer : list of str
        Noms des paramètres à inférer.
    borne_min, borne_max : torch.Tensor
        Bornes des priors.
    MODEL_NN : str
        Modèle de densité pour posterior_nn (ex. "maf", "nsf").
    EXP_NAME : str
        Nom de l'expérience (sert de préfixe pour sauvegarde).
    NUM_SIMULATION : int
        Nombre de simulations à effectuer.
    NUM_SAMPLES : int
        Nombre d'échantillons postérieurs à tirer.
    REAL_OBS : bool
        Si True, utilise une observation réelle (pas de true_theta).
    save_dir : str
        Dossier dans lequel les résultats sont sauvegardés.
    """

    # Créer le prior
    prior = BoxUniform(low=borne_min, high=borne_max)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Équivalent en gaussienne (si on veut une normale multivariée) :

    # mean = (borne_max + borne_min) / 2  # centre de la gaussienne
    # std = (borne_max - borne_min) / 4   # écart-type raisonnable (ajuster selon besoin)
    # cov = torch.diag(std ** 2)          # matrice de covariance diagonale
    # prior = MultivariateNormal(mean, cov)

    # Préparation du simulateur pour sbi
    simulator = process_simulator(simulator_func, prior, prior_returns_numpy)

    # Simulation
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=NUM_SIMULATION, num_workers=-1, seed=0)
    theta = theta.to(torch.float32)
    x = x.to(torch.float32)

    # Entraînement SBI
    density_estimator = posterior_nn(model=MODEL_NN)
    inference = NPE(prior=prior, density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)

    # Observation
    if REAL_OBS:
        _, FC_obs = generate_observation()
        theta_obs = None
    else:
        theta_dict, FC_obs = generate_observation(params=params_obs, list_inferer=list_inferer)
        theta_obs = torch.tensor([theta_dict[p] for p in list_inferer], dtype=torch.float32)

    print(f" Observation générée. FC_obs.shape = {FC_obs.shape}")

    # Sauvegarde de la posterior
    os.makedirs(f"{save_dir}/posteriors", exist_ok=True)
    save_density_and_obs(density_estimator, FC_obs, f"{save_dir}/posteriors/posterior_{EXP_NAME}.pt")

    # Inférence
    theta_posterior, x_predictive_all = compute_predictions_from_posterior(
        posterior, FC_obs, simulator_func, list_inferer, N_samples=NUM_SAMPLES
    )

    # Logging & visualisation
    log_experiment_results(
        true_theta=theta_obs,
        posterior_samples=theta_posterior,
        predictions=x_predictive_all,
        observation=FC_obs,
        borne_min=borne_min,
        borne_max=borne_max,
        posterior_nn_model=MODEL_NN,
        experiment_name=EXP_NAME,
        list_inferer=list_inferer,
        time_elapsed=default_params['run_time'],
        num_samples=NUM_SAMPLES,
        num_simulations=NUM_SIMULATION,
        output_dir=save_dir,
        file_name="experiences.json"
    )

def multi_round_inference(
    simulator_func = simulator_FC,
    params_obs = params_obs,
    list_inferer = list_inferer,
    borne_min = borne_min,
    borne_max = borne_max,
    FC_obs=None,
    REAL_OBS=REAL_OBS,
    NUM_ROUNDS=NUM_ROUNDS,
    NUM_SIMULATION=NUM_SIMULATION,
    NUM_SAMPLES=NUM_SAMPLES,
    MODEL_NN=MODEL_NN,
    EXP_NAME=EXP_NAME,
    save_dir="results"
):
    """
    Effectue une inférence SBI non-amortie en plusieurs rounds, ciblée sur une seule observation (x₀).

    Chaque round affine le postérieur autour de cette observation, pour une estimation plus précise.

    Paramètres
    ----------
    simulator_func : callable
        Fonction de simulation (e.g. simulator_FC).
    params_obs : dict
        Paramètres par défaut utilisés pour simuler l'observation.
    list_inferer : list of str
        Paramètres à inférer.
    borne_min, borne_max : torch.Tensor
        Bornes du prior.
    FC_obs : torch.Tensor or None
        Observation (optionnel si REAL_OBS=False, sinon générée).
    REAL_OBS : bool
        Indique si l'observation est simulée ou issue de données réelles.
    NUM_ROUNDS : int
        Nombre de rounds SBI (au moins 2).
    NUM_SIMULATION : int
        Simulations par round.
    NUM_SAMPLES : int
        Échantillons tirés à la fin depuis la postérieure.
    MODEL_NN : str
        Architecture de la postérieure (e.g. 'maf', 'nsf').
    EXP_NAME : str
        Nom de l'expérience (utilisé pour les fichiers).
    save_dir : str
        Répertoire de sauvegarde.
    """

    prior = BoxUniform(low=borne_min, high=borne_max)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator_func, prior, prior_returns_numpy)

    if FC_obs is None:
        if REAL_OBS:
            _, FC_obs = generate_observation()
            theta_obs = None
        else:
            theta_dict, FC_obs = generate_observation(params=params_obs, list_inferer=list_inferer)
            theta_obs = torch.tensor([theta_dict[p] for p in list_inferer], dtype=torch.float32)
    else:
        theta_obs = None

    print('startig inference')
    inference = NPE(prior=prior, density_estimator=posterior_nn(model=MODEL_NN))
    proposal = prior
    posteriors = []

    for round_idx in range(NUM_ROUNDS):
        print(f"\n🔁 Round {round_idx+1}/{NUM_ROUNDS}")
        theta, x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=NUM_SIMULATION, num_workers=-1, seed=0)
        theta = theta.to(torch.float32)
        x = x.to(torch.float32)
        inference = inference.append_simulations(theta, x, proposal=proposal)
        density_estimator = inference.train(show_train_summary=True)
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(FC_obs)

    # Inférence conditionnelle sur l’observation
    theta_posterior, x_predictive_all = compute_predictions_from_posterior(
        posterior, FC_obs, simulator_func, list_inferer, N_samples=NUM_SAMPLES
    )

    # Logging & visualisation
    log_experiment_results(
        true_theta=theta_obs,
        posterior_samples=theta_posterior,
        predictions=x_predictive_all,
        observation=FC_obs,
        borne_min=borne_min,
        borne_max=borne_max,
        posterior_nn_model=MODEL_NN,
        experiment_name=EXP_NAME,
        list_inferer=list_inferer,
        time_elapsed=default_params['run_time'],
        num_samples=NUM_SAMPLES,
        num_simulations=NUM_SIMULATION * NUM_ROUNDS,
        output_dir=save_dir,
        file_name="experiences.json"
    )


################## MAIN ######################

if __name__ == "__main__":
    # Exemple d'utilisation
    single_round_inference()
    #multi_round_inference()



