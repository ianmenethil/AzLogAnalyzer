#regexManager.property
import logging
from pathlib import Path
from pandas import isna
from typing import List, Dict
from configLoader import ConfigLoader, regexConfigurator

# config_dict = ConfigLoader.load_configurations()
logger = logging.getLogger(__name__)
import csv


class RegexManager:

    @staticmethod
    def filter_df_based_on_patterns(df, config_dict, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_cols = df.columns.tolist()
        all_cols_row_count = df.shape[0]
        all_cols_col_count = df.shape[1]
        save_pattern_rows = []
        for column, patterns in config_dict.items():
            for index, pattern in enumerate(patterns):
                match_type = 'match' if pattern.startswith("^") and pattern.endswith("$") else 'contains'

                if match_type == 'match':
                    pattern_filter = df[column].str.match(pattern, na=False)
                    logger.info(f'Match: {column}_{index} - {pattern} - {pattern_filter}')
                    save_pattern_rows.append(f'Match: {column}_{index} - {pattern} - {pattern_filter}')
                else:
                    pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
                    logger.warning(f'Contains: {column}_{index} - {pattern} - {pattern_filter}')
                    save_pattern_rows.append(f'Contains: {column}_{index} - {pattern} - {pattern_filter}')
                # Ensure no NaN values in the pattern_filter
                pattern_filter = pattern_filter.fillna(False)
                if not pattern_filter.any():
                    logger.warning(f"No matches found for {column}_{index} - {pattern}")
                    save_pattern_rows.append(f"No matches found for {column}_{index} - {pattern}")
                else:
                    logger.info(f"Found {len(pattern_filter[pattern_filter == True])} matches for {column}_{index} - {pattern}")
                    save_pattern_rows.append(f"Found {len(pattern_filter[pattern_filter == True])} matches for {column}_{index} - {pattern}")
                matched_rows = df[pattern_filter]
                if matched_rows.empty:
                    logger.warning(f"No matched rows found for {column}_{index} - {pattern}")
                    save_pattern_rows.append(f"No matched rows found for {column}_{index} - {pattern}")
                else:
                    logger.info(f"Found {len(matched_rows)} matched rows for {column}_{index} - {pattern}")
                    save_pattern_rows.append(f"Found {len(matched_rows)} matched rows for {column}_{index} - {pattern}")

                if not matched_rows.empty:
                    matched_filename = f"{output_dir}/{column}_{index}_matched_rows.csv"
                    matched_rows.to_csv(matched_filename, index=False)
                    logger.info(f"Saved {len(matched_rows)} matched rows to {matched_filename}")

                    df = df[~pattern_filter]
                    all_cols_after_filter = df.columns.tolist()
                    save_pattern_rows.append(f"Saved {len(matched_rows)} matched rows to {matched_filename}")
                    save_pattern_rows.append(f'Original columns: {all_cols}')
                    save_pattern_rows.append(f'After filter columns: {all_cols_after_filter}')
                    filter_drop_col_c = df.shape[1]
                    filter_drop_row_c = df.shape[0]
                    save_pattern_rows.append(f'Difference between all_cols_row_count and filter_drop_row_c: {all_cols_col_count - filter_drop_col_c}')
                    logger.info(f'Difference between all_cols_row_count and filter_drop_row_c: {all_cols_col_count - filter_drop_col_c}')
                    after_drop_cols = None  # Declare the variable before using it
                    logger.info(f'Difference between filter_drop_row_c and filter_drop_col_c: {all_cols_row_count - filter_drop_row_c}')
                    df = df.dropna(axis=1, how='all')
                    after_drop_col_c = df.shape[1]
                    after_drop_row_c = df.shape[0]
                    logger.info(f'Difference between all_cols_row_count and filter_drop_row_c: {all_cols_col_count - after_drop_col_c}')
                    logger.info(f'Difference between filter_drop_row_c and filter_drop_col_c: {all_cols_row_count - after_drop_row_c}')
                    save_pattern_rows.append(f'After Dropped columns: {after_drop_cols}')
                    save_pattern_rows.append(f'Difference between all_cols_row_count and after_drop_row_c: {all_cols_col_count - after_drop_col_c}')
                    save_pattern_rows.append(f'Difference between after_drop_row_c and after_drop_col_c: {all_cols_row_count - after_drop_row_c}')

                    after_drop_cols = df.columns.tolist()
                    after_drop_cols = set(all_cols) - set(after_drop_cols)
                    save_pattern_rows.append(f'After Dropped columns: {after_drop_cols}')
                    logger.info(f'Original columns: {all_cols}')
                    logger.info(f'After filter columns: {all_cols_after_filter}')
                    logger.info(f'After Dropped columns: {after_drop_cols}')
                    with open('temp.csv', 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        save_pattern_rows = list(set(save_pattern_rows))
                        writer.writerow(save_pattern_rows)
                    csvfile.close()
        return df
