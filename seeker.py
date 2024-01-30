import logging
import os
from rich.logging import RichHandler
import pandas as pd

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
                    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_path=False)])
logger = logging.getLogger(__name__)


def save_data_to_excel(data, output_file):
    if data is not None:
        data.to_excel(output_file, index=False)
    else:
        logger.error("No data to save, received None.")


# Example usage
column_key_pairs = {'CsUriStem': 'requestUri_s', 'CsUriQuery': 'requestUriQuery_s', 'CIp': 'IPs', 'UserAgent': 'userAgent_s'}


def find_and_save_matches(dataframe1, dataframe2, col_key_pairs):
    for key, value in col_key_pairs.items():
        # Check for the existence of the column in both dataframes
        if key not in dataframe1.columns:
            print(f"Column {key} not found in dataframe1. Skipping this pair.")
            continue
        if value not in dataframe2.columns:
            print(f"Column {value} not found in dataframe2. Skipping this pair.")
            continue

        # Proceed with matching since columns exist
        matched_rows_df1 = dataframe1[dataframe1[key].isin(dataframe2[value])]
        matched_rows_df2 = dataframe2[dataframe2[value].isin(dataframe1[key])]

        # Combining matched rows from both dataframes
        combined_matched_rows = pd.concat([matched_rows_df1, matched_rows_df2])

        # Save the combined matches to a file
        filename = f'match_{key}_vs_{value}.xlsx'
        combined_matched_rows.to_excel(filename, index=False)
        print(f"Saved matches for {key} vs {value} to {filename}")

        return combined_matched_rows


def load_excel(filename):
    try:
        return pd.read_excel(filename)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return None


# # Define UserAgent parsing patterns
def main():
    # delete output.xlsx summary_output.xlsx
    if os.path.exists('matches.xlsx'):
        os.remove('matches.xlsx')
    if os.path.exists('match_summary.xlsx'):
        os.remove('match_summary.xlsx')
    # Load full DataFrames, not just columns
    az_diag_df = load_excel('azdiag.xlsx')
    # print all columnns from the excel file to the cli immidiately after loading
    http_logs_df = load_excel('httplogs.xlsx')
    data = find_and_save_matches(http_logs_df, az_diag_df, column_key_pairs)
    if data is not None:
        save_data_to_excel(data, 'matches.xlsx')
    else:
        logger.error("Matching process failed, no data to save.")
    return data


main()
