import pandas as pd
import numpy as np

Prefrontal_dict = {'FRP': ['FRP', 'FRP1', 'FRP2/3', 'FRP5', 'FRP6a', 'FRP6b'],
                   'ACAd': ['ACAd', 'ACAd1', 'ACAd2/3', 'ACAd5', 'ACAd6a', 'ACAd6b'],
                   'ACAv': ['ACAv', 'ACAv1', 'ACAv2/3', 'ACAv5', 'ACAv6a', 'ACAv6b'],
                   'PL': ['PL', 'PL1', 'PL2', 'PL2/3', 'PL5', 'PL6a', 'PL6b'],
                   'ILA': ['ILA', 'ILA1', 'ILA2', 'ILA2/3', 'ILA5', 'ILA6a', 'ILA6b'],
                   'ORBl': ['ORBl', 'ORBl1', 'ORBl2/3', 'ORBl5', 'ORBl6a', 'ORBl6b'],
                   'ORBm': ['ORBm', 'ORBm1', 'ORBm2', 'ORBm2/3', 'ORBm5', 'ORBm6a', 'ORBm6b'],
                   'ORBvl': ['ORBvl', 'ORBvl1', 'ORBvl2/3', 'ORBvl5', 'ORBvl6a', 'ORBvl6b']} 

Lateral_dict = {'AId': ['AId', 'AId1', 'AId2/3', 'AId5', 'AId6a', 'AId6b'], 
                'AIv': ['AIv', 'AIv1', 'AIv2/3', 'AIv5', 'AIv6a', 'AIv6b'], 
                'AIp': ['AIp', 'AIp1', 'AIp2/3', 'AIp5', 'AIp6a', 'AIp6b'], 
                'GU': ['GU', 'GU1', 'GU2/3', 'GU4', 'GU5', 'GU6a', 'GU6b'], 
                'VISC': ['VISC', 'VISC1', 'VISC2/3', 'VISC4', 'VISC5', 'VISC6a', 'VISC6b']}

Somatomotor_dict = {'SSs': ['SSs', 'SSs1', 'SSs2/3', 'SSs4', 'SSs5', 'SSs6a', 'SSs6b'], 
               'SSp-bfd': ['SSp-bfd', 'SSp-bfd1', 'SSp-bfd2/3', 'SSp-bfd4', 'SSp-bfd5', 'SSp-bfd6a', 'SSp-bfd6b'],
               'SSp-tr': ['SSp-tr', 'SSp-tr1', 'SSp-tr2/3', 'SSp-tr4', 'SSp-tr5', 'SSp-tr6a', 'SSp-tr6b'], 
               'SSp-ll': ['SSp-ll', 'SSp-ll1', 'SSp-ll2/3', 'SSp-ll4', 'SSp-ll5', 'SSp-ll6a', 'SSp-ll6b'], 
               'SSp-ul': ['SSp-ul', 'SSp-ul1', 'SSp-ul2/3', 'SSp-ul4', 'SSp-ul5', 'SSp-ul6a', 'SSp-ul6b'], 
               'SSp-un': ['SSp-un', 'SSp-un1', 'SSp-un2/3', 'SSp-un4', 'SSp-un5', 'SSp-un6a', 'SSp-un6b'], 
               'SSp-n': ['SSp-n', 'SSp-n1', 'SSp-n2/3', 'SSp-n4', 'SSp-n5', 'SSp-n6a', 'SSp-n6b'], 
               'SSp-m': ['SSp-m', 'SSp-m1', 'SSp-m2/3', 'SSp-m4', 'SSp-m5', 'SSp-m6a', 'SSp-m6b'], 
               'MOp': ['MOp', 'MOp1', 'MOp2/3', 'MOp5', 'MOp6a', 'MOp6b'], 
               'MOs': ['MOs', 'MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b']}

Visual_dict = {'VISal': ['VISal', 'VISal1', 'VISal2/3', 'VISal4', 'VISal5', 'VISal6a', 'VISal6b'], 
               'VISl': ['VISl', 'VISl1', 'VISl2/3', 'VISl4', 'VISl5', 'VISl6a', 'VISl6b'], 
               'VISp': ['VISp', 'VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b'], 
               'VISli': ['VISli', 'VISli1', 'VISli2/3', 'VISli4', 'VISli5', 'VISli6a', 'VISli6b'], 
               'VISrl': ['VISrl', 'VISrl1', 'VISrl2/3', 'VISrl4', 'VISrl5', 'VISrl6a', 'VISrl6b']}


Medial_dict = {'VISa': ['VISa', 'VISa1', 'VISa2/3', 'VISa4', 'VISa5', 'VISa6a', 'VISa6b'], 
               'VISam': ['VISam', 'VISam1', 'VISam2/3', 'VISam4', 'VISam5', 'VISam6a', 'VISam6b'], 
               'VISpm': ['VISpm', 'VISpm1', 'VISpm2/3', 'VISpm4', 'VISpm5', 'VISpm6a', 'VISpm6b'], 
               'RSPagl': ['RSPagl', 'RSPagl1', 'RSPagl2/3', 'RSPagl5', 'RSPagl6a', 'RSPagl6b'], 
               'RSPd': ['RSPd', 'RSPd1', 'RSPd2/3', 'RSPd4', 'RSPd5', 'RSPd6a', 'RSPd6b'], 
               'RSPv': ['RSPv', 'RSPv1', 'RSPv2', 'RSPv2/3', 'RSPv5', 'RSPv6a', 'RSPv6b']}

Aud_dict = {'AUDd': ['AUDd', 'AUDd1', 'AUDd2/3', 'AUDd4', 'AUDd5', 'AUDd6a', 'AUDd6b'], 
            'AUDp': ['AUDp', 'AUDp1', 'AUDp2/3', 'AUDp4', 'AUDp5', 'AUDp6a', 'AUDp6b'], 
            'AUDpo': ['AUDpo', 'AUDpo1', 'AUDpo2/3', 'AUDpo4', 'AUDpo5', 'AUDpo6a', 'AUDpo6b'], 
            'AUDv': ['AUDv', 'AUDv1', 'AUDv2/3', 'AUDv4', 'AUDv5', 'AUDv6a', 'AUDv6b']}

Thalamus_dict = {'VENT': ['VENT', 'PoT', 'VAL', 'VM', 'VP', 'VPL', 'VPLpc', 'VPM', 'VPMpc'], 
                 'GENd': ['GENd', 'LGd', 'LGd-sh', 'LGd-co', 'LGd-ip', 'MG', 'MGd', 'MGv', 'MGm'], 
                 'SPF':[ 'SPF', 'SPFm', 'SPFp'],
                 'SPA':['SPA'],
                 'PP':['PP'],
                 'LAT': ['LAT', 'Eth', 'LP', 'PO', 'POL', 'REth', 'SGN'],
                 'ATN': ['ATN', 'AD', 'AM', 'AMd', 'AMv', 'AV', 'IAD', 'IAM', 'LD'], 
                 'MED': ['MED', 'IMD', 'MD', 'MDc', 'MDl', 'MDm', 'PR', 'SMT'], 
                 'MTN': ['MTN', 'RE', 'PT', 'PVT', 'Xi'], 
                 'ILM': ['ILM', 'CL', 'CM', 'PCN', 'PF', 'PIL', 'RH'], 
                 'RT': ['RT'], 
                 'GENv': ['GENv', 'IGL', 'IntG', 'SubG', 'LGv', 'LGvl', 'LGvm'], 
                 'EPI': ['EPI', 'LH', 'MH', 'PIN']}


Midbrain_dict = {
 'MBmot': ['MBmot', 'SNr', 'VTA', 'PN', 'RR', 'MRN', 'MRNm', 'MRNmg', 'MRNp', 'SCm', 'SCdg', 'SCdw', 'SCiw', 'SCig', 'SCig-a', 'SCig-b', 'SCig-c', 'PAG', 'INC', 'ND', 'PRC', 'Su3', 'PRT', 'APN', 'MPT', 'NOT', 'NPC', 'OP', 'PPT', 'RPF', 'InCo', 'CUN', 'RN', 'III', 'MA3', 'EW', 'IV', 'Pa4', 'VTN', 'AT', 'LT', 'DT', 'MT', 'SNl']}


## Striatum_dict
Striatum_dict = {'STRd': ['STRd', 'CP'],
           'STRv': ['STRv', 'ACB', 'FS', 'OT', 'isl', 'islm', 'OT1', 'OT2', 'OT3'],
           'LSX': ['LSX', 'LS', 'LSc', 'LSr', 'LSv', 'SF', 'SH']}

## Pallidum_dict
Pallidum_dict = {'PALd': ['PALd', 'GPe', 'GPi'],
           'PALv': ['PALv', 'SI', 'MA'],
           'PALm': ['PALm', 'MSC', 'MS', 'NDB', 'TRS'],
           'PALc': ['PALc', 'BST', 'BSTa', 'BSTal', 'BSTam', 'BSTdm', 'BSTfu', 'BSTju', 'BSTmg', 'BSTov', 'BSTrh', 'BSTv', 'BSTp', 'BSTd', 'BSTpr', 'BSTif', 'BSTtr', 'BSTse', 'BAC']}


## CTXsp 
CTXsp_dict={
        'CLA': ['CLA'],
        }

## Hippocampal_dict
Hippocampal_dict={
         'CA1': ['CA1', 'CA1slm', 'CA1so', 'CA1sp', 'CA1sr'],
         'CA2': ['CA2', 'CA2slm', 'CA2so', 'CA2sp', 'CA2sr'],
         'CA3': ['CA3', 'CA3slm', 'CA3slu', 'CA3so', 'CA3sp', 'CA3sr'],
         'DG': ['DG', 'DG-mo', 'DG-po', 'DG-sg', 'DG-sgz', 'DGcr', 'DGcr-mo', 'DGcr-po', 'DGcr-sg', 'DGlb', 'DGlb-mo', 'DGlb-po', 'DGlb-sg', 'DGmb', 'DGmb-mo', 'DGmb-po', 'DGmb-sg'],
         'POST': ['POST', 'POST1', 'POST2', 'POST3'],
        'SUBd': ['SUB', 'SUBd', 'SUBd-m', 'SUBd-sp', 'SUBd-sr'],
        'SUBv': ['SUBv', 'SUBv-m', 'SUBv-sp', 'SUBv-sr'],
        'PRE': ['PRE', 'PRE1', 'PRE2', 'PRE3'],
}


All_dict = {**Prefrontal_dict, **Lateral_dict, **Somatomotor_dict, **Visual_dict, **Medial_dict, **Aud_dict, **Thalamus_dict, **Midbrain_dict, **Striatum_dict, **Pallidum_dict, **CTXsp_dict, **Hippocampal_dict}



CCF_info = pd.read_csv('/data100/dataset/mice_2021/Allen_CCF/P56_Label.csv')
CCF_index = np.concatenate([CCF_info['Index'].values, CCF_info['Index'].values + 10000])
CCF_index[CCF_index == 10000] = 0
CCF_new = pd.DataFrame({
    'index': CCF_index,
    'ACR_name': np.concatenate([CCF_info['ACR_name'].values + '_lh', CCF_info['ACR_name'].values + '_rh']),
    'Full_name': np.concatenate([CCF_info['Full_name'].values + '_lh', CCF_info['Full_name'].values + '_rh'])
}).set_index('Full_name')