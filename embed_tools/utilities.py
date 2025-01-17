from typing import Optional, Callable

def print_features_table(features_dict: dict[str, int], title: Optional[str] = None):
    """
    Prints a dictionary of features and their counts in a table with box characters.

    Args:
        features_dict (dict): A dictionary where keys are feature names and values are their counts.
        title (str, optional): Title for the table. Defaults to None.
    """
    # Determine column widths
    key_col_width = max(len(str(key)) for key in features_dict.keys()) + 2
    value_col_width = max(len(f"{value:,}") for value in features_dict.values()) + 2

    # Define table characters
    _chars = {
        "horz": "─", "vert": "│", "top_l": "┌", "top_r": "┐",
        "bot_l": "└", "bot_r": "┘", "mid_l": "├", "mid_r": "┤",
        "top_mid": "┬", "bot_mid": "┴", "mid_mid": "┼",
    }

    # Calculate table width
    table_width = key_col_width + value_col_width + 3

    # Print title if specified
    if title:
        print(f" {title.center(table_width, ' ')} ")

    # Print top border
    print(f"{_chars['top_l']}{_chars['horz'] * (key_col_width + 1)}{_chars['top_mid']}{_chars['horz'] * (value_col_width + 1)}{_chars['top_r']}")

    # Print header
    header = f"{_chars['vert']} {'Feature':<{key_col_width - 1}} {_chars['vert']} {'Count':<{value_col_width - 1}} {_chars['vert']}"
    print(header)

    # Print header divider
    print(f"{_chars['mid_l']}{_chars['horz'] * (key_col_width + 1)}{_chars['mid_mid']}{_chars['horz'] * (value_col_width + 1)}{_chars['mid_r']}")

    # Print each feature and count
    for feature, count in features_dict.items():
        row = f"{_chars['vert']} {feature:<{key_col_width - 1}} {_chars['vert']} {count:<{value_col_width - 1},} {_chars['vert']}"
        print(row)

    # Print bottom border
    print(f"{_chars['bot_l']}{_chars['horz'] * (key_col_width + 1)}{_chars['bot_mid']}{_chars['horz'] * (value_col_width + 1)}{_chars['bot_r']}")
