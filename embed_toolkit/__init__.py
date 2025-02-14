"""
embed_toolkit: A Python package for processing and analyzing data from the Emory Breast Imaging Dataset (EMBED).

This package provides tools for working with Pandas DataFrames from the EMBED dataset, 
including utilities for extracting, summarizing, and correcting findings. The toolkit is designed 
to streamline the analysis of the data by centralizing commonly used functions.

Modules:
    - `magview`: Contains core classes and functions for EMBED data processing.
    - `utilities`: General utility functions for print formatting, type checking, etc.

Exports:
    - `EMBEDParameters`: A class for managing configuration parameters, including column lists 
      and summary functions, used throughout the toolkit.
    - `EMBEDDataFrameTools`: A Pandas DataFrame accessor providing methods for interacting 
      with EMBED-formatted data, accessible via `.embed.METHOD`.
    - `correct_contralaterals`: A standalone function for ensuring that missing contralateral 
      findings are included in the data when required.

Example Usage:
    # Import the package
    from EMBED_toolkit import EMBEDParameters, EMBEDDataFrameTools, correct_contralaterals

    # Use the `.embed` accessor to apply methods to a DataFrame
    df.embed.head_cols()
    df.embed.summarize(title="Summary of Findings")

    # Apply the `correct_contralaterals` function to a DataFrame
    corrected_df = correct_contralaterals(df)

Dependencies:
    - pandas
    - tqdm
    - numpy

"""

from .magview import EMBEDParameters, EMBEDDataFrameTools # tool classes
from .magview import correct_contralaterals # functions

# TODO: automate tests to warn about breaking changes/ensure core functionality
# TODO: decide top-level user interface points which to be retained between versions (only these top-level interfaces should be imported in this file)