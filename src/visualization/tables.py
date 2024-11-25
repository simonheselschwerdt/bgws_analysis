"""
src/visualization/tables.py

This script provides functions visualize and save tables.

Functions:
- save_tabel

Usage:
    Import this module in your scripts.
"""

import os
import pandas as pd

def save_tabel(df, save_dir, filename):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)    

    print(f'Table saved under {filepath}')