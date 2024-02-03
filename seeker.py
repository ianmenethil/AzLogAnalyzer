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
    logger.info("Starting matching process.")
    logger.info(f"Datafrae {dataframe1} and dataframe {dataframe2}{col_key_pairs}.")

    for key, value in col_key_pairs.items():
        # Check for the existence of the column in both dataframes
        if key not in dataframe1:
            logger.info(f"Column {key} not found in dataframe1. Skipping this pair.")
            continue
        if value not in dataframe2:
            logger.info(f"Column {value} not found in dataframe2. Skipping this pair.")
            continue

        # Proceed with matching since columns exist
        matched_rows_df1 = dataframe1[dataframe1[key].isin(dataframe2[value])]
        matched_rows_df2 = dataframe2[dataframe2[value].isin(dataframe1[key])]

        # Combining matched rows from both dataframes
        combined_matched_rows = pd.concat([matched_rows_df1, matched_rows_df2])

        # Save the combined matches to a file
        filename = f'match_{key}_vs_{value}.xlsx'
        combined_matched_rows.to_excel(filename, index=False)
        logger.info(f"Saved matches for {key} vs {value} to {filename}")

        return combined_matched_rows


def load_excel(filename):
    try:
        return pd.read_excel(filename)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return None


def main():
    os.chdir(os.path.dirname(__file__))
    compare_folder = 'compare/'
    matches_file = 'matches.xlsx'
    summary_file = 'match_summary.xlsx'
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
    os.chdir(compare_folder)
    file1 = load_excel(f'{compare_folder}appreq.xlsx')
    file2 = load_excel(f'{compare_folder}AzDiag.xlsx')
    # check if both files exist
    if os.path.exists(str(file1)) and os.path.exists(str(file2)):
        logger.info("Both files exist.")
    else:
        logger.info("One or both files do not exist.")

    # Check if 'Time' column exists in both dataframes
    if file1 is not None and 'TimeGenerated' not in file1.columns:
        logger.info("Column 'TimeGenerated' not found in azdiag.xlsx. Cannot proceed with matching.")
        return None
    if file2 is not None and 'TimeGenerated' not in file2.columns:
        logger.info("Column 'TimeGenerated' not found in httplogs.xlsx. Cannot proceed with matching.")
        return None
    logger.info("Column 'TimeGenerated' found in both files.")
    # Perform matching based on 'Time' column
    matched_data = find_and_save_matches(file2, file1, {'TimeGenerated': 'TimeGenerated'})

    if matched_data is not None:
        save_data_to_excel(matched_data, 'matched_records.xlsx')
        logger.info("Matching process completed. Matched records saved to matched_records.xlsx.")
    else:
        logger.error("Matching process failed, no data to save.")

    return matched_data


main()
