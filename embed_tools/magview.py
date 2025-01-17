"""
Tools to process clinical data in EMBED.
"""
from pandas import DataFrame, Series, concat
from numpy import nan
from tqdm.auto import tqdm
from pandas.api.extensions import register_dataframe_accessor

# utilities defines utility functions (like print formatting) and basic types/type imports
# TODO: modify to remove relative imports
from .utilities import print_features_table, Optional, Callable


class EMBEDParameters:
    """
    A class to contain parameters for EMBED dataframes
    """
    # a dictionary containing lists of feature names that are unique at the patient, exam, finding, and procedure levels
    level_columns: dict[str, list[str]] = {
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
    head_columns: list[str] = ['empi_anon', 'acc_anon', 'study_date_anon', 'desc', 'side', 'asses', 'path_severity', 'bside', 'procdate_anon']

    
    def __init__(self, is_open_data: bool = False, is_anon: bool = True) -> None:
        self.is_open_data: bool = is_open_data
        self.is_anon: bool = is_anon

        # TODO: make this work with Open Data/Non-Open Data versions
        # TODO: make this work with Anon/Raw versions [NOTE: public versions must not expose PHI column names etc.]
        # TODO: should static default count methods be kept in here for inheritance or moved outside the class?
        # TODO: allow EMBEDParameters to be modified globally by the user to change all downstream usage? how should this be done?

    @staticmethod
    def count_patients(df: DataFrame) -> int:
        try:
            n = df['empi_anon'].nunique()
        except KeyError:
            n = -9 # -9 indicates the feature is missing
        return n
    
    @staticmethod
    def count_exams(df: DataFrame) -> int:
        try:
            n = df['acc_anon'].nunique()
        except KeyError:
            n = -9 # -9 indicates the feature is missing
        return n
    
    @staticmethod
    def count_findings(df: DataFrame) -> int:
        try:
            n = (df['acc_anon'].astype(str) + "_" + df['numfind'].astype(str)).nunique()
        except KeyError:
            n = -9 # -9 indicates the feature is missing
        return n
    
    @staticmethod
    def count_pngs(df: DataFrame) -> int:
        try:
            n = df['png_path'].nunique()
        except KeyError:
            n = -9 # -9 indicates the feature is missing
        return n

    @staticmethod
    def count_dicoms(df: DataFrame) -> int:
        try:
            n = df['anon_dicom_path'].nunique()
        except KeyError:
            n = -9 # -9 indicates the feature is missing
        return n

    def summary_count(self, df: DataFrame, summary_dict: Optional[dict[str, Callable]] = None) -> dict[str, int]:
        """
        Generates a summary of counts for specified features in a DataFrame.

        This method calculates counts for various features (e.g., patients, exams, findings, 
        PNGs, DICOMs) in the provided DataFrame using functions specified in `summary_dict`. 
        If no `summary_dict` is provided, a default set of counting functions is used.

        Args:
            df (DataFrame): The input DataFrame containing the data to summarize.
            summary_dict (Optional[dict[str, Callable]]): A dictionary where keys are feature names 
                (e.g., "Patients", "Exams") and values are functions that take the DataFrame as input 
                and return a count for the corresponding feature. If `summary_dict` is None, the 
                following default functions are used:
                - "Patients": `self.count_patients`
                - "Exams": `self.count_exams`
                - "Findings": `self.count_findings`
                - "PNGs": `self.count_pngs`
                - "DICOMs": `self.count_dicoms`

        Returns:
            dict[str, int]: A dictionary where keys are feature names and values are their respective counts.

        Example Usage:
            # Use default summary functions
            counts = embed_params.summary_count(df)

            # Use custom summary functions
            custom_summary_dict = {
                "CustomFeature": custom_count_function,
            }
            counts = embed_params.summary_count(df, summary_dict=custom_summary_dict)

        Notes:
            - Functions in `summary_dict` should return -9 to indicate missing features. 
            Such features are skipped in the returned dictionary.
        """
        if summary_dict is None:
            # use default summary count methods
            summary_dict: dict[str, Callable] = {
                "Patients": self.count_patients,
                "Exams": self.count_exams,
                "Findings": self.count_findings,
                "PNGs": self.count_pngs,
                "DICOMs": self.count_dicoms,
            }

        count_dict = dict()
        for feature_name, count_func in summary_dict.items():
            feature_count = count_func(df)
            # skip missing features
            if feature_count != -9:
                count_dict[feature_name] = feature_count
        return count_dict

    def list_columns(self, levels: Optional[list[str]] = None) -> list[str]:
        """
        Lists column names associated with specified levels of EMBED hierarchy.

        This method retrieves the column names corresponding to the levels of the EMBED hierarchy, 
        which include 'patient', 'exam', 'finding', and 'procedure'. If no levels are specified, 
        it returns column names for all levels.

        Args:
            levels (Optional[list[str]]): A list of strings specifying the levels of the EMBED hierarchy 
                for which to retrieve column names. Valid levels are:
                - 'patient'
                - 'exam'
                - 'finding'
                - 'procedure'
                If `levels` is None, column names for all levels will be returned.

        Returns:
            list[str]: A list of column names corresponding to the specified levels.

        Example Usage:
            # List columns for specific levels
            columns = embed_params.list_columns(levels=['exam', 'finding'])
            
            # List columns for all levels
            all_columns = embed_params.list_columns()

        Notes:
            - The levels and their associated columns are defined in `self.level_columns`, 
            which is expected to be a dictionary where keys are level names and values 
            are lists of column names.
        """
        if levels is None:
            # if levels is undefined, return columns for all levels
            levels: list[str] = list(self.level_columns.keys())

        out_list: list[str] = []
        for level in levels:
            out_list.extend(self.level_columns[level])

        return out_list

    @staticmethod
    def extract_exam_laterality(row: Series) -> str:
        """
        Extracts the laterality of an exam from its description.

        This method checks the description of an exam to derive its laterality. 
        It returns one of the following values based on the presence of keywords in the 
        description:
        - 'B' for bilateral findings (if 'bilat' is included in the description).
        - 'L' for left-sided findings (if 'left' is included).
        - 'R' for right-sided findings (if 'right' is included).
        - 'None' if none of the above conditions were true

        Args:
            row (Series): A row from a Pandas DataFrame containing the exam description 
                in the 'desc' column.

        Returns:
            str: A string representing the laterality of the exam. Possible values are 
                'B', 'L', 'R', or 'None'.

        Example Usage:
            # Apply the method to a DataFrame to extract laterality for each row
            df["exam_laterality"] = df.apply(embed_params.extract_exam_laterality, axis=1)

        Notes:
            - The string matching is case-insensitive.
        """
        finding_desc = row.desc.lower()
        
        if ("bilat" in finding_desc):
            return "B"
        elif ("left" in finding_desc):
            return "L"
        elif ("right" in finding_desc):
            return "R"
        else:
            return "None"

    @staticmethod
    def aggregate_birads(group: DataFrame) -> str:
        """
        Aggregates finding BIRADS assessment for an exam to get the most severe.

        Args:
            group (DataFrame): A DataFrame containing findings for a single exam. 
                The DataFrame must include the following columns:
                - 'desc': Exam description.
                - 'asses': Findings BIRADS assessments.

        Returns:
            str: The BIRADS category corresponding to the worst assessment in the group. 
                Possible values are 'A', 'B', 'N', 'P', 'S', 'M', or 'K'. If no valid 
                assessment is found, an empty string is returned.

        Example Usage:
            # Group the DataFrame by exam ID
            # extract a dict containing the exam ID -> worst BIRADS mapping for each exam
            exam_br_dict = df.groupby('acc_anon').apply(embed_params.aggregate_birads).to_dict()

            # Map aggregated BIRADS scores back to the DataFrame on exam IDs
            df['exam_birads'] = df['acc_anon'].map(exam_br_dict)

        """
        exam_desc: str = group.desc.iloc[0].item()
        is_screen_exam: bool = "screen" in exam_desc.lower()

        if is_screen_exam:
            br_to_val_dict: dict[str, int] = {
                'A': 0, # 'A' maps to birads 0
                'B': 1, # 'B' maps to birads 2
                'N': 2  # 'N' maps to birads 1
            }
        else:
            br_to_val_dict: dict[str, int] = {
                'N': 5, # 'N' maps to birads 1
                'B': 4, # 'B' maps to birads 2
                'P': 3, # 'P' maps to birads 3
                'S': 2, # 'S' maps to birads 4
                'M': 1, # 'M' maps to birads 5
                'K': 0  # 'K' maps to birads 6
            }
        
        # map birads scores against finding assessments, then get the worst (min) int
        worst_br_val: int = min(group.asses.map(br_to_val_dict).tolist())
        # flip the br_to_val_dict to convert it back to a string (empty string if invalid)
        val_to_br_dict: dict[int, str] = {v:k for k,v in br_to_val_dict.items()}
        worst_br_str: str = val_to_br_dict.get(worst_br_val, '')
        return worst_br_str

    @staticmethod
    def extract_characteristics(row: Series) -> dict[str, int]:
        """
        Extracts finding-level imaging characteristics from a given row of data.

        This method processes a pandas Series representing a single row of data
        and determines the presence (1) or absence (0) of specific imaging 
        features (Mass, Asymmetry, Architectural Distortion, Calcification).

        The output is a dictionary with binary values indicating the presence of these features.

        Example Usage:
            mag_df[['mass', 'asymmetry', 'arch_distortion', 'calcification']] = mag_df.apply(
                embed_params.extract_finding_characteristics, 
                axis='columns', 
                result_type='expand'
            )

        Args:
            row (Series): A pandas Series object representing a single row of data. 
                The Series should contain the following keys:
                - 'massshape'
                - 'massmargin'
                - 'massdens'
                - 'calcdistri'
                - 'calcfind'
                - 'calcnumber'

        Returns:
            dict: A dictionary with keys ['mass', 'asymmetry', 'arch_distortion', 'calcification'],
            where each key maps to 0 (absent) or 1 (present).
        """
        
        # output imaging features coded as either 0: absent or 1: present
        findings_dict: dict[str, int] = {
            'mass': 0,
            'asymmetry': 0,
            'arch_distortion': 0,
            'calcification': 0
        }
    
        if ((row['massshape'] in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']) 
            or (row['massmargin'] in ['D', 'U', 'M', 'I', 'S']) 
            or (row['massdens'] in ['+', '-', '='])):
            findings_dict['mass'] = 1
    
        if row['massshape'] in ['T', 'B', 'S', 'F', 'V']:
            findings_dict['asymmetry'] = 1
    
        if row['massshape']in ['Q', 'A']:
            findings_dict['arch_distortion'] = 1
    
        if ((row['calcdistri'] is not nan) 
            or (row['calcfind'] is not nan) 
            or (row['calcnumber'] > 0)):
            findings_dict['calcification'] = 1
    
        return findings_dict

    def aggregate_characteristics(self, group: DataFrame):
        """
        Aggregates finding-level characteristics to the exam-level.

        This method processes a dataframe grouped by `acc_anon`, where each group represents 
        findings from a unique exam, and aggregates the characteristics to determine
        their presence (1) or absence (0) at the exam level. If any finding in the group
        has a characteristic present (value of 1), the exam-level characteristic is marked
        as present (1).

        Example Usage:
            exam_characteristics: dict[int, dict[str, bool]] = mag_df.groupby('acc_anon').apply(
                embed_params.aggregate_characteristics
            ).to_dict()
            
            for char_type in ['exam_mass', 'exam_asymmetry', 'exam_arch_distortion', 'exam_calcification']:
                char_dict = {k: v[char_type] for k, v in exam_characteristics.items()}
                mag_df[char_type] = mag_df['acc_anon'].map(char_dict)

        Args:
            group (DataFrame): A pandas DataFrame representing a group of findings
                from an exam. The DataFrame must contain the following columns:
                - 'mass': Binary column indicating the presence (1) or absence (0) of a mass.
                - 'asymmetry': Binary column indicating the presence (1) or absence (0) of an asymmetry.
                - 'arch_distortion': Binary column indicating the presence (1) or absence (0) of an architectural distortion.
                - 'calcification': Binary column indicating the presence (1) or absence (0) of a calcification.

        Returns:
            dict: A dictionary with keys ['exam_mass', 'exam_asymmetry', 'exam_arch_distortion', 'exam_calcification'],
            where each key maps to 1 if the corresponding characteristic is present in any finding
            within the group, or 0 otherwise.

        Notes:
            - This method assumes that finding-level characteristics have already been extracted
            (e.g., using `extract_characteristics`).
        """
        return {
            'exam_mass': int(any(group.mass)),
            'exam_asymmetry': int(any(group.asymmetry)),
            'exam_arch_distortion': int(any(group.arch_distortion)),
            'exam_calcification': int(any(group.calcification))
        }


@register_dataframe_accessor("embed")
class EMBEDDataFrameTools:
    """
    A class containing custom methods to interact with an EMBED Pandas DataFrame.

    This class provides tools for working with dataframes from the EMBED dataset. 
    The methods are accessible by calling `.embed.METHOD` on a Pandas DataFrame.

    Attributes:
        _params (EMBEDParameters): An instance of the `EMBEDParameters` class, 
            used to manage configuration and parameter settings for EMBED operations.

    Methods:
        head_cols(*cols, col_list=None):
            Displays a subset of key columns from the DataFrame, optionally sorted by date.
        summarize(title=None, print_counts=True):
            Provides a summary of the data, including counts of key features.

    Example Usage:
        # Access methods via the `.embed` accessor
        df.embed.head_cols('column1', 'column2')
        df.embed.summarize(title="Data Summary")
    """
    # instantiate an EMBED params object as self._params
    _params: EMBEDParameters = EMBEDParameters()

    def __init__(self, pandas_object):
        self._df = pandas_object

    def head_cols(self, *cols: str, col_list: Optional[list[str]] = None) -> DataFrame:
        """
        Returns a subset of the DataFrame with commonly used or user-specified columns.

        This method retrieves a subset of the DataFrame using a default set of 
        commonly used columns, optionally extended or overridden by user-specified 
        columns. If the `study_date_anon` column is available, the subset is sorted 
        by this column.

        Args:
            *cols (str): Additional column names to include in the subset.
            col_list (Optional[list[str]]): A custom list of column names to use. 
                If provided, it overrides the default column list.

        Returns:
            DataFrame: A subset of the DataFrame containing the specified columns.

        Example Usage:
            # Retrieve a subset of the DataFrame with default and additional columns
            subset = df.embed.head_cols('extra_col1', 'extra_col2')

            # Retrieve a subset using a custom column list
            subset = df.embed.head_cols(col_list=['col1', 'col2', 'col3'])
        """
        # returns a subset of the dataframe using a set of commonly used minimum columns (can be overwritten by specifying `col_list` or added to with positional string args)
        if col_list is None:
            default_col_list = self._params.head_columns
            col_list = [c for c in [*default_col_list, *cols] if c in self._df.columns]

        try:
            # return df sorted by date if possible
            return self._df[col_list].sort_values('study_date_anon')

        except KeyError:
            # otherwise just return the df
            return self._df[col_list]

    def summarize(self, title: Optional[str] = None, print_counts: bool = True) -> None:
        count_dict = self._params.summary_count(self._df)
        if print_counts:
            print_features_table(count_dict, title)
        else:
            return count_dict


def correct_contralaterals(df: DataFrame, derived_finding_code: int = -9) -> DataFrame:
    """
    Ensures that negative contralateral findings are included for exams that imply their presence.

    This function processes a DataFrame of findings to identify exams that require the addition 
    of contralateral findings when they are missing. It creates new rows for the contralateral 
    side with derived finding codes, ensuring consistency with the original data structure.

    Args:
        df (DataFrame): The input DataFrame containing exam and finding data. 
            It must include the following columns:
            - 'side': Specifies the side of the finding ('L', 'R', 'B', or NaN).
            - 'desc': Description of the exam, used to identify bilateral exams.
            - 'acc_anon': Exam-level identifier column.
            - 'study_date_anon': Date of the study, used for sorting.
        derived_finding_code (int, optional): The code to assign to the `numfind` column 
            for derived rows. Defaults to -9.

    Returns:
        DataFrame: A corrected DataFrame that includes derived contralateral findings 
        for exams that implied their presence but lacked explicit entries. Side can be ('L', 'R', or 'B')

    Example Usage:
        corrected_df = correct_contralaterals(original_df)

    Notes:
        - The function identifies exams that are bilateral (`desc` contains 'bilat') 
          but lack findings for both sides ('L' or 'R').
        - For such exams, it determines the missing side and creates a new row with 
          derived values for that side.
        - The `asses` column is set to 'N' and `numfind` is set to the specified `derived_finding_code` 
          for the derived rows.
        - The output DataFrame is sorted by `study_date_anon` and the index is reset.

    Steps:
        1. Normalize the `side` column to treat empty strings or NaN as 'B' (bilateral).
        2. Identify bilateral exams (`desc` contains 'bilat').
        3. Identify exams with bilateral findings ('side' == 'B')
        4. Identify bilateral exams with no bilateral bilateral findings, 
           then get the list of unique L/R finding sides
        5. For each of these exams, if they have only 1 unique finding side, get the 
           contralateral side and add them to a dict of exams to correct
        6. Create derived rows with cloned patient/exam level columns
        7. Combine the original and derived rows, sort by `study_date_anon`, and reset the index.

    """
    embed_params: EMBEDParameters = EMBEDParameters()
    out_df = df.copy() # copy the dataframe to ensure we don't modify the original
    
    # numfind for all derived rows will be coded as specified
    # create a list to track the columns that should be copied to derived rows
    col_copy_list: list[str] = []
    
    # extract all exam and patient level features
    col_copy_list = embed_params.list_columns(['exam', 'patient'])

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

    # initialize correction_df
    correction_df = DataFrame(data=None, columns=out_df.columns, index=range(n_correction))
    
    for i, (acc, correction_side) in tqdm(enumerate(acc_contralat_correction_dict.items()), total=n_correction):
        # take the first row associated with the same acc and extract the column details to copy over 
        # (only add columns to the copy list that are consistent for all rows in each exam)
        copy_dict = {col_name:col_val for col_name,col_val in out_df[out_df.acc_anon == acc].iloc[0].to_dict().items() if col_name in col_copy_list}
        
        # update the values that are constant for all contralateral corrections
        copy_dict.update({"asses": "N", "side": correction_side, "numfind": derived_finding_code})

        # copy the information in the dict to the correction df at the current index
        correction_df.iloc[i] = copy_dict
        
    # finally, concat the output and correction dfs, then sort by study date and reset the index
    return concat([out_df, correction_df]).sort_values(['empi_anon', 'acc_anon', 'numfind']).reset_index(drop=True)
