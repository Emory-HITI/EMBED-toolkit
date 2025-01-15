# tools to work with magview in the Emory Breast Imaging Dataset (EMBED)
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from tqdm.auto import tqdm

# utilities defines utility functions (like print formatting) and basic types/type imports
from .utilities import print_features_table, Optional


class EMBEDMagviewTools:
    # a dictionary containing lists of feature names that are unique at the patient, exam, finding, and procedure levels
    _level_features: dict[str, list[str]] = {
        "patient": [
            'GENDER_DESC','ETHNICITY_DESC','ETHNIC_GROUP_DESC',
            'MARITAL_STATUS_DESC','ENCOUNTER_QTY','empi_anon','cohort_num',
        ],
        "exam": [
            'loc_num','tech_init','init','proccode','desc','vtype',
            'tissueden','case','age_at_study','acc_anon','study_date_anon',
            'sdate_anon','first_3_zip','total_L_find','total_R_find',
        ],
        "finding": [
            'massshape','massmargin','massdens','calcfind','calcdistri',
            'calcnumber','otherfind','implanfind','consistent','side',
            'size','location','depth','distance','numfind','asses',
            'recc','stable','new','changed',
        ],
        "procedure": [
            'type','technique','biopsite','biop_loc','bcomp',
            'path_loc','diag_out','surgery','lymphsurg','surg_loc',
            'pocomp','ltcomp','bside','path1','path2',
            'path3','path4','path5','path6','path7','path8',
            'path9','path10','concord','hgrade','tnmpt','tnmpn',
            'tnmm','tnmdesc','tnmr','stage','loc','bdepth',
            'bdistance','focality','nfocal','specsize','specsize2',
            'specsize3','dcissize','invsize','superior','inferior',
            'anterior','posterior','medial','lateral','specinteg',
            'specnum','specembed','est','estp','her2','fish',
            'ki67','extracap','methodevl','snode_rem','node_rem',
            'node_pos','macrometa','micrometa','isocell','largedp',
            'eic','procdate_anon','pdate_anon','path_group',
            'path_severity',
        ],
    }
    # a minimal list of features to use when inspecting rows from magview
    head_features: list[str] = ['empi_anon', 'acc_anon', 'study_date_anon', 'desc', 'side', 'asses', 'path_severity', 'bside', 'procdate_anon']

    # default id column names
    patient_id: str = 'empi_anon'
    exam_id: str = 'acc_anon'
    image_paths: list[str] = ["png_path", "anon_dicom_path"]
    
    def __init__(self, is_open_data: bool = False, is_anon: bool = True):
        self.is_open_data: bool = is_open_data
        self.is_anon: bool = is_anon

    def level_features(self, levels: list[str]):
        out_list: list[str] = []
        for level in levels:
            out_list.extend(self._level_features[level])

        return out_list

    def extract_finding_characteristics(self, row: pd.Series):
        # example usage:
        # mag_df[['mass', 'asymmetry', 'arch_distortion', 'calcification']] = mag_df.apply(
        #     mag_tools.extract_finding_characteristics, 
        #     axis='columns', 
        #     result_type='expand'
        # )
        
        # output imaging features coded as either 0: absent or 1: present
        findings_dict = {
            'mass': 0,
            'asymmetry': 0,
            'arch_distortion': 0,
            'calcification': 0
        }
    
        if (row['massshape'] in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']) or\
        (row['massmargin'] in ['D', 'U', 'M', 'I', 'S']) or\
        (row['massdens'] in ['+', '-', '=']):
            findings_dict['mass'] = 1
    
        if row['massshape'] in ['T', 'B', 'S', 'F', 'V']:
            findings_dict['asymmetry'] = 1
    
        if row['massshape']in ['Q', 'A']:
            findings_dict['arch_distortion'] = 1
    
        if (row['calcdistri'] is not np.nan) or\
        (row['calcfind'] is not np.nan) or\
        (row['calcnumber'] > 0):
            findings_dict['calcification'] = 1
    
        return findings_dict

    def generalize_exam_characteristics(self, group: pd.DataFrame):
        # example_usage:
        # exam_characteristics: dict[int, dict[str, bool]] = mag_df.groupby('acc_anon').apply(mag_tools.generalize_exam_characteristics).to_dict()
        # for char_type in ['exam_mass', 'exam_asymmetry', 'exam_arch_distortion', 'exam_calcification']:
        #     char_dict = {k:v[char_type] for k,v in exam_characteristics.items()}
        #     mag_df[char_type] = mag_df['acc_anon'].map(char_dict)
        
        # TODO: output bools as ints instead to be consistent with self.extract_finding_characteristics ?
    
        # extract_finding_characteristics() must have been run on the data previously to
        # generate finding-level characteristics
        # return True if the characteristic is True on any of its findings
        return {
            'exam_mass': any(group.mass),
            'exam_asymmetry': any(group.asymmetry),
            'exam_arch_distortion': any(group.arch_distortion),
            'exam_calcification': any(group.calcification)
        }


@register_dataframe_accessor("embed")
class EMBEDDataFrameTools:
    _mag_tools = EMBEDMagviewTools()
    _anon_col_dict = {
        "patients": _mag_tools.patient_id,
        "exams": _mag_tools.exam_id,
        "images": _mag_tools.image_paths,
    }

    def __init__(self, pandas_object):
        self._df = pandas_object

    def head_cols(self, *cols: str, col_list: Optional[list[str]] = None) -> None:
        # prints a set of commonly used minimum columns (can be overwritten by specifying `col_list` or added to with positional string args)
        if col_list is None:
            default_col_list = self._mag_tools.head_features
            col_list = [c for c in [*default_col_list, *cols] if c in self._df.columns]

        try:
            # display sorted by date if possible
            display(self._df[col_list].sort_values('study_date_anon'))

        except KeyError:
            # otherwise just display
            display(self._df[col_list])

    def inspect(self, title: Optional[str] = None, anon: bool = True, col_dict: Optional[dict] = None) -> None:
        # use default columns unless specified
        if anon and col_dict is None:
            col_dict = self._anon_col_dict
            
        elif col_dict is None: # anon == False
            raise NotImplementedError("the default column dict for non-anon data has not been specified yet.")

        count_dict = dict()
        for col_type, col_name in col_dict.items():
            # for col types with a list of names, check all and take the max n unique
            if isinstance(col_name, list):
                col_type, col_n = self._inspect_list_col(col_type, col_name)
                
            else:
                col_n = self._df[col_name].nunique()
                
            if col_type != "": # skip missing columns
                count_dict[col_type] = col_n
            
        print_features_table(count_dict, title)

    def _inspect_list_col(self, col_type: str, col_list: list[str]) -> tuple[str, int]:
        n_list = []
        for cname in col_list:
            try:
                n = self._df[cname].nunique()
                n_list.append(n)
            except KeyError:
                pass
                
        if n_list:
            col_n = max(n_list)
        elif col_type == 'images':
            # special case for list of image cols, we should check findings instead
            col_type = "findings"
            col_n = len(set(self._df['acc_anon'].astype(str) + "_" + self._df['numfind'].astype(str)))
        else:
            # if none of the relevant columns existed, skip this column type
            return ("", 0)
            
        return (col_type, col_n)


def correct_contralaterals(df: pd.DataFrame, derived_finding_code: int = -9.0):
    # function to correct the dataframe to ensure that negative contralateral findings are included
    # for exams that imply their presence
    out_df = df.copy() # copy the dataframe to ensure we don't modify the original
    
    # numfind for all derived rows will be coded as specified
    # create a list to track the columns that should be copied to derived rows
    col_copy_list: list[str] = []
    
    # import an EMBEDMagviewTools object and extract all exam and patient level features
    mag_tools = EMBEDMagviewTools()
    col_copy_list = mag_tools.level_features(['exam', 'patient'])

    # get list of exams that require contralateral correction
    # normalize 'side' column: treat empty string/nan as 'B'
    out_df['side'] = out_df['side'].replace('', 'B').fillna('B')
    bilat_acc_list = out_df[out_df.desc.str.contains('bilat', case=False)].acc_anon.unique().tolist()
    
    # get the number of exams with bilateral findings
    b_find_list = out_df[out_df.side == "B"].acc_anon.unique().tolist()
    
    # select the bilateral exams with no "B" findings, then 
    # get the number of unique "L"/"R" lateralities for each exam as a dict
    exam_finding_unique_sides_dict = out_df[
        out_df.acc_anon.isin(bilat_acc_list) 
        & ~out_df.acc_anon.isin(b_find_list) 
        & out_df.side.isin(['L', 'R'])
    ].groupby('acc_anon').side.unique().to_dict()

    # get the list of these exams needing correction and find the contralateral side to add
    acc_contralat_correction_dict = {acc:("L" if sides[0] == "R" else "R" ) for acc,sides in exam_finding_unique_sides_dict.items() if len(sides) == 1}
    n_correction = len(acc_contralat_correction_dict)
    
    # make a copy of the structure of the df to store the new correction columns in
    # create a dictionary of column names and their corresponding dtypes from out_df
    # dtypes_dict = {col:dtype for col,dtype in zip(out_df.columns, out_df.dtypes)}

    # initialize correction_df with the same dtypes as out_df
    # correction_df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dtypes_dict.items()}, index=range(n_correction))
    correction_df = pd.DataFrame(data=None, columns=out_df.columns, index=range(n_correction))
    
    for i, (acc, correction_side) in tqdm(enumerate(acc_contralat_correction_dict.items()), total=n_correction):
        # take the first row associated with the same acc and extract the column details to copy over 
        # (only add columns to the copy list that are consistent for all rows in each exam)
        copy_dict = {col_name:col_val for col_name,col_val in out_df[out_df.acc_anon == acc].iloc[0].to_dict().items() if col_name in col_copy_list}
        
        # update the values that are constant for all contralateral corrections
        copy_dict.update({"asses": "N", "side": correction_side, "numfind": derived_finding_code})

        # copy the information in the dict to the correction df at the current index
        correction_df.iloc[i] = copy_dict
        
    # finally, concat the output and correction dfs, then sort by study date and reset the index
    return pd.concat([out_df, correction_df]).sort_values("study_date_anon").reset_index(drop=True)#.astype(out_df.dtypes)
