from enum import Enum


class ProcedureType(Enum):
    BIOPSY = "B"
    SURGERY = "S"


class Procedure:
    def __init__(self): 
        self.type: ProcedureType

"""
"procedure": [
    "type", -- string -- "proc_type"
    "technique", -- string -- "proc_technique"
    "biopsite", -- string -- "proc_biopsy_site"
    "biop_loc", -- int -- "proc_biopsy_loc"
    "bcomp", -- string -- "proc_bcomp"
    "path_loc", -- int -- "proc_path_loc"
    "diag_out", -- string/bool (currently N/Y/NaN) -- "proc_diag_out" -- what is this???
    "surgery", -- string -- "proc_surgery"
    "lymphsurg", -- string -- "proc_lymph_surgery"
    "surg_loc", -- int -- "proc_surgery_loc"
    "pocomp", -- all NaN?
    "ltcomp", -- all NaN?
    "bside", -- string -- "proc_side"
    "path1", -- string
    "path2", -- string
    "path3", -- string
    "path4", -- string
    "path5", -- string
    "path6", -- string
    "path7", -- string
    "path8", -- string
    "path9", -- string
    "path10", -- string
    "concord", -- string -- "path_concord"
    "hgrade", -- string -- "path_hgrade"
    "tnmpt", -- string -- "path_tnm_pt"
    "tnmpn", -- string -- "path_tnm_pn"
    "tnmm", -- string -- "path_tnm_m"
    "tnmdesc", -- string -- "path_tnm_desc"
    "tnmr", -- all NaN? -- "path_tnm_r"
    "stage", -- string -- "path_stage"
    "bdepth", -- string -- "proc_biopsy{?}_depth"
    "bdistance", -- int -- "proc_biopsy{?}_distance"
    "focality", -- string -- "path_focality"
    "nfocal", -- int -- "path_n_focal"
    "specsize", -- float -- "path_spec_size"
    "specsize2", -- float -- "path_spec_size2"
    "specsize3", -- float -- "path_spec_size3"
    "dcissize", -- float -- "path_dcis_size"
    "invsize", -- float -- "path_inv_size"
    "superior", -- float -- "path_superior" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "inferior", -- float -- "path_inferior" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "anterior", -- float -- "path_anterior" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "posterior", -- float -- "path_posterior" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "medial", -- float -- "path_medial" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "lateral", -- float -- "path_lateral" -- CHECK IF THIS IS ACTUALLY FOR PROCEDURES
    "specinteg", -- string -- "path_spec_integ"
    "specnum", -- int -- "path_spec_num"
    "specembed", -- string -- "path_spec_embed"
    "est", -- string -- "path_est"
    "estp", -- string -- "path_estp"
    "her2", -- string -- "path_her2"
    "fish", -- string -- "path_fish"
    "ki67", -- string -- "path_ki67"
    "extracap", -- string/bool (Y/N/NaN) -- "path_extra_cap"
    "methodevl", -- string -- "path_method_evl"
    "snode_rem", -- float -- "path_snode_rem"
    "node_rem", -- float -- "path_node_rem"
    "node_pos", -- float -- "path_node_pos"
    "macrometa", -- float -- "path_macrometa"
    "micrometa", -- float -- "path_micrometa"
    "isocell", -- float -- "path_isocell"
    "largedp", -- float -- "path_largedp"
    "eic", -- string -- "path_eic"
    "procdate_anon", -- datetime -- "proc_date_anon"
    "pdate_anon", -- datetime -- "pdate_anon"
    "path_group", -- int
    "path_severity", -- int
],%                                                                                                                                                                                                                                          
"""
