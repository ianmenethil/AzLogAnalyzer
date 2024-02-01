import logging
from typing import List, Dict
from pathlib import Path
from configLoader import ConfigLoader, regexConfigurator

config_dict = ConfigLoader.load_configurations()
logger = logging.getLogger(__name__)


class RegexManager:

    @staticmethod
    def filter_df_based_on_patterns(df, config_dict, output_dir):
        """
        Filters the DataFrame based on patterns and saves matched rows to separate CSV files.

        Parameters:
            df (pd.DataFrame): The DataFrame to filter.
            config (dict): Configuration dict containing columns and their respective patterns.
            output_dir (str): Directory where matched row CSVs will be saved.
        """
        logger.info(f'Filtering DataFrame based on patterns in {config_dict}')
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

        for column, patterns in config_dict['MATCH_VALUE_TO_SPECIFIC_COLUMN'].items():
            for index, pattern in enumerate(patterns):
                # Determine if the pattern is for a full match or partial match
                match_type = 'match' if pattern.startswith("^") and pattern.endswith("$") else 'contains'

                # Apply the pattern filter based on the match type
                if match_type == 'match':
                    pattern_filter = df[column].str.match(pattern)
                else:  # 'contains'
                    pattern_filter = df[column].str.contains(pattern, na=False, regex=True)

                # Extract matched rows
                matched_rows = df[pattern_filter]

                # Save matched rows to a CSV if any exist
                if not matched_rows.empty:
                    matched_rows.to_csv(f"{output_dir}/{column}_{index}_matched_rows.csv", index=False)
                    logger.info(f"Saved {len(matched_rows)} matched rows to {output_dir}/{column}_{index}_matched_rows.csv")

                    # Optionally, remove matched rows from the original DataFrame
                    df = df[~pattern_filter]

        return df

    # @staticmethod
    # def remove_patterns(dataframe, extraction_file, regex, string, key_col_to_val_patts) -> Any:
    #     """The `remove_patterns` function takes a DF, extraction file, regex patterns, string patterns,
    #     and key-value patterns as input, removes the specified patterns from the DF, and returns the
    #     modified DF."""
    #     try:
    #         df = dataframe
    #         columns_to_search = [KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestQuery_s, KEYSEARCH_requestUri_s]
    #         total_removed = 0
    #         output = ""
    #         df_columns = df.columns.tolist()
    #         df_total_columns = len(df.columns)
    #         df_total_rows = len(df)
    #         logger.info(f'DF Total Columns: {df_total_columns} - Total Rows: {df_total_rows}')
    #         logger.info(f'df_columns {df_columns}')
    #         regex_json_formatted = json.dumps(regex, indent=4)
    #         string_json_formatted = json.dumps(string, indent=4)
    #         logger.info('\nRegex patterns:')
    #         logger.info(regex_json_formatted)
    #         logger.info('\nString patterns:')
    #         logger.info(string_json_formatted)
    #         try:
    #             df = RegexManager.remove_regex_patterns(df, regex, string, columns_to_search, extraction_file)
    #         except Exception as e:
    #             logger.error(f'Error in remove_regex_patterns: {e}', exc_info=True, stack_info=True)
    #         try:
    #             df = RegexManager.remove_key_value_patterns(df, key_col_to_val_patts, extraction_file)
    #         except Exception as e:
    #             logger.error(f'Error in remove_key_value_patterns: {e}', exc_info=True, stack_info=True)
    #         output += f"\nTotal rows Total Removed: {total_removed}:\nRemaining Rows: {len(df)}\n"
    #         return df
    #     except Exception as e:
    #         logger.error(f'Error in remove_patterns: {e}', exc_info=True, stack_info=True)

    # @staticmethod
    # def remove_patterns(df, patterns, pattern_type, log_file):
    #     """Removes rows based on regex or key-value patterns and logs the changes."""
    #     total_removed = 0
    #     columns_to_search = [KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestQuery_s, KEYSEARCH_requestUri_s]
    #     for column in columns_to_search:
    #         if column not in df.columns:
    #             logger.info(f'Column {column} not found in the DF.')
    #             continue
    #         for pattern in patterns:
    #             if pattern_type == 'regex':
    #                 pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
    #             else:  # Assume key-value pattern matching
    #                 pattern_filter = df[column].isin(pattern)

    #             removed_rows = df[pattern_filter]
    #             removed_count = len(removed_rows)
    #             if removed_count > 0:
    #                 logger.info(f'Pattern {pattern} - Column {column} - Removed Rows {removed_count}')
    #                 LogManager.save_removed_rows_to_raw_logs(removed_rows, log_file)
    #                 df = df[~pattern_filter]
    #                 total_removed += removed_count
    #     return df

    # @staticmethod
    # def process_patterns(df, extraction_file, regex, strings, key_col_to_val_patts):
    #     """Processes both regex and key-value patterns."""
    #     columns_to_search = [KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestQuery_s, KEYSEARCH_requestUri_s]
    #     df = RegexManager.remove_patterns(df, regex_list, regex, extraction_file)
    #     for key, val_patterns in key_col_to_val_patts.items():
    #         df = RegexManager.remove_patterns(df, string_list, strings, extraction_file)
    #     return df

    # @staticmethod
    # def processExclusionPairs(df, exclusion_pairs, log_file):
    #     """Applies exclusion rules to the DataFrame and logs removed rows."""
    #     for column, values_to_exclude in exclusion_pairs.items():
    #         if column in df.columns:
    #             filter_condition = df[column].isin(values_to_exclude)
    #             removed_rows = df[filter_condition].copy()
    #             if not removed_rows.empty:
    #                 LogManager.save_removed_rows_to_raw_logs(removed_rows, log_file)
    #             df = df[~filter_condition]
    #     return df

    # @staticmethod
    # def find_duplicate_columns(df):
    #     """Identifies and returns a list of duplicate column names in the DataFrame."""
    #     duplicate_columns = df.columns[df.columns.duplicated()]
    #     return duplicate_columns.tolist()

    # @staticmethod
    # def remove_regex_patterns(df, regex, string, columns_to_search, extraction_file) -> Any:
    #     """The `remove_regex_patterns` function removes rows from a DF that match specified regex
    #     patterns in specified columns and saves the removed rows to a file."""
    #     total_removed = 0
    #     output = ""
    #     for pattern in regex + string:
    #         logger.info(f'Looking for pattern: {pattern} in columns: {columns_to_search}')
    #         for column in columns_to_search:
    #             if column not in df.columns:
    #                 logger.info(f'Column {column} not found in the DF.')
    #                 continue
    #             pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
    #             removed_rows = df[pattern_filter]

    #             logger.info(f'REGEX Pat {pattern} - Col {column} - Removed Rows {len(removed_rows)}')
    #             logger.info(f'Rowdata: removed_rows[column]: {removed_rows[column]}')

    #             if not removed_rows.empty:
    #                 LogManager.save_removed_rows_to_raw_logs(removed_rows, extraction_file)
    #                 output += f"Pattern: {pattern} - Column: {column} - Removed Count: {len(removed_rows)}"
    #                 removed_rows_text = removed_rows[column]
    #                 output += str(removed_rows_text)
    #             df = df[~pattern_filter]
    #             removed_count = len(removed_rows)
    #             total_removed += removed_count
    #     return df

    # @staticmethod
    # def remove_key_value_patterns(df, key_col_to_val_patts, extraction_file):
    #     """The `remove_key_value_patterns` function removes rows from a DF based on specified key-value patterns in specific columns and logs the removed rows."""
    #     logger.info('[yellow]##########Key Value Removals Started##########[/yellow]')
    #     total_removed = 0
    #     output = ""
    #     for column, patterns in key_col_to_val_patts.items():
    #         removed_count = 0  # ! Initialize removed_count at the start of the loop
    #         logger.info(f'Looking for column: {column} patterns {patterns} key_cl.items {key_col_to_val_patts.items()}')
    #         if column not in df.columns:
    #             logger.warning(f'Column {column} not found in the DF.')
    #             continue
    #         logger.info(f"Processing column: {column} with patterns: {patterns}")
    #         for pattern in patterns:
    #             pattern_filter = df[column].str.contains(pattern, na=False, regex=True)
    #             removed_rows = df[pattern_filter]
    #             if not removed_rows.empty:
    #                 removed_count = len(removed_rows)
    #                 logger.info('###KEY_CL_TO_PATT Removals###')
    #                 logger.info(f'Pat {pattern} - Col {column} - Removed Rows {removed_count}')
    #                 LogManager.save_removed_rows_to_raw_logs(removed_rows, extraction_file)
    #                 df = df[~pattern_filter]
    #                 total_removed += removed_count
    #             if removed_count > 0:
    #                 output += f"\nMatch:Pattern: {pattern} \nRemoved Count: {removed_count} \nColumn: {column}\n"
    #             else:
    #                 output += f"\nNo match:\nPattern: {pattern} \nColumn: {column}\n"
    #     logger.info('[yellow]##########Key Value Removals Completed##########[/yellow]')
    #     return df

    # @staticmethod
    # def processExclusionPairs(df: pd.DataFrame, filename: str, exclusion_pairs, log_file: str):
    #     try:
    #         for column, values_to_exclude in exclusion_pairs.items():
    #             if column in df.columns:
    #                 logger.info(f"Column: {column} found in DF. Values to exclude: {values_to_exclude}")
    #                 filter_condition = df[column].isin(values_to_exclude)
    #                 # ! Capture the rows that will be removed
    #                 removed_rows = df[filter_condition].copy()  # ? Use .copy() to avoid SettingWithCopyWarning
    #                 # ? Log the removed rows if there are any
    #                 if not removed_rows.empty:
    #                     pattern = f"Exclusion: {values_to_exclude}"
    #                     LogManager.save_removed_rows_to_raw_logs(removed_rows, log_file)
    #                     logger.info(f"Removed rows based on column '{column}' and pattern '{pattern}' have been logged to {log_file}")
    #                 # ? update df to exclude the rows
    #                 df = df[~filter_condition]
    #                 if df.empty:
    #                     logger.info("DataFrame is empty after exclusion.")
    #                     break  # ! Exit the loop if df is empty
    #                 # ! Identify columns that may be dropped due to becoming entirely NaN
    #                 columns_before = set(df.columns)
    #                 df.dropna(axis=1, how='all')
    #                 # df.dropna(axis=1, how='all', inplace=True)
    #                 columns_after = set(df.columns)
    #                 # columns_dropped = columns_before - columns_after  # ? NOTE
    #                 # if columns_dropped:
    #                 if columns_dropped := columns_before - columns_after:
    #                     logger.info(f"Columns dropped because they became empty after exclusion: {columns_dropped}")
    #                 else:
    #                     logger.warning("No columns were dropped after exclusion.")
    #         # ! After processing, save the potentially modified df
    #         if not df.empty:
    #             df.to_csv(filename, index=False)
    #             logger.info(f"Modified DF saved to {filename}")
    #         else:
    #             logger.warning("DataFrame is empty after exclusions. Skipping saving to CSV.")
    #         return df
    #     except Exception as e:
    #         logger.error(f"Error in exclusion processing: {e}", exc_info=True)
    #         return df

    # @staticmethod
    # def find_duplicate_columns(df) -> Any:
    #     """The function `find_duplicate_columns` takes a DF as input and returns a list of duplicate
    #     column names."""
    #     duplicate_columns = df.columns[df.columns.duplicated()]
    #     # logger.info(f"Duplicate columns: {duplicate_columns.tolist()}")
    #     return duplicate_columns.tolist()
