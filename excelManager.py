import logging
from typing import List, Dict, Any
import pandas as pd  # pylint: disable=C0411

logger = logging.getLogger(__name__)


class ExcelManager():

    @staticmethod
    def excelCreate(input_csv_file, output_excel_file,) -> None:
        try:
            df = pd.read_csv(input_csv_file, low_memory=False)
            df = df.dropna(axis=1, how='all')
            if not df.empty:
                with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:  # pylint: disable=abstract-class-instantiated
                    df.to_excel(writer, index=False, sheet_name='Sheet1')
                    ExcelManager.format_excel_file(writer, df)
                    logger.info(f'Excel formatted and created: {output_excel_file}')
                    return output_excel_file
            else:
                logger.warning('DF is empty')
        except Exception as e:
            logger.error(f'E in create_excel: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def format_excel_file(writer, df) -> None:
        """The `format_excel_file` function formats an Excel file by setting the font, font size, and column width for each column in a given DF."""
        try:
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            cell_format = workbook.add_format({'font_name': 'Open Sans', 'font_size': 8})
            worksheet.set_column('A:ZZ', None, cell_format)
            for column in df:
                column_length = max(df[column].astype(str).apply(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.set_column(col_idx, col_idx, column_length, cell_format)
        except Exception as e:
            logger.error(f'Error in format_excel_file: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    @staticmethod
    def create_final_excel(input_file: str, output_file: str, logfile: str, columns_to_be_dropped: List[str], final_column_order: List[str]):
        def concatenate_values(row: pd.Series, columns: List[str], include_col_names: bool = False) -> str:
            if include_col_names:
                return ', '.join([f'{col}: "{row[col]}"' for col in columns if pd.notna(row[col])])
            return ', '.join([f'"{row[col]}"' for col in columns if pd.notna(row[col])])

        def drop_and_create_columns(df: pd.DataFrame, new_columns: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
            logger.info('[red]########### columns_to_drop started ###########[/red]')
            for col_name, info in new_columns.items():
                existing_cols = [col for col in info['cols'] if col in df.columns]
                missing_cols = [col for col in info['cols'] if col not in df.columns]
                if existing_cols:
                    # ! Create new column by concatenating values from existing columns
                    df[col_name] = df.apply(lambda row, cols=existing_cols: concatenate_values(row, cols, info['include_names']), axis=1)  # pylint: disable=W0640
                    logger.info(f'New column "{col_name}" created with values from columns: {existing_cols} - Total rows: {df.shape[0]}')
                    if col_name != 'Qs':
                        columns_to_drop_due_to_concat.extend(existing_cols)
                    else:
                        logger.info(f'Columns kept: {existing_cols}')
                if missing_cols:
                    logger.info(f'Columns not found in "{col_name}": {missing_cols}')
            if columns_to_drop_due_to_concat:
                # ! Drop columns that were used to create new columns
                df.drop(columns=columns_to_drop_due_to_concat, inplace=True)
                dropped_columns = available_columns_before_drop - set(df.columns)
                logger.info(f'Columns dropped from concat: {columns_to_drop_due_to_concat}')
                if dropped_columns:
                    logger.info("Columns dropped from concat:")
                    for column in dropped_columns:
                        logger.info(column)
                else:
                    logger.info("No columns were dropped.")
            logger.info('[red]########### columns_to_drop  from concat completed ###########[/red]')
            logger.info(f'Columns dropped.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
            logger.info(f'[green]Columns to be dropped started: {columns_to_be_dropped}[/green]')
            available_columns_before_drop2 = set(df.columns)
            cols_to_drop_from_func_def = [column for column in columns_to_be_dropped if column in df.columns]
            # ! Drop columns specified in the function definition
            if cols_to_drop_from_func_def:
                df.drop(columns=cols_to_drop_from_func_def, inplace=True)
                dropped_columns2 = available_columns_before_drop2 - set(df.columns)
                if dropped_columns2:
                    logger.info("Columns dropped:")
                    for column in dropped_columns2:
                        logger.info(column)
                logger.info(f'Columns dropped 2.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
                logger.info('[green]Columns to be dropped ended: [/green]')
            return df

        def apply_final_column_order_and_save(df: pd.DataFrame, final_column_order: List[str], output_file: str, available_columns_before_drop: set) -> None:
            final_order_logs = []
            final_order_logs += ['Final order started']
            final_order_logs += ['actual_final_order']
            actual_final_order = [col for col in final_column_order if col in df.columns]
            final_order_logs += [str(actual_final_order)]
            final_order_logs += ['specified_columns']
            specified_columns = set(actual_final_order)
            final_order_logs += [str(specified_columns)]
            final_order_logs += ['unspecified_columns']
            unspecified_columns = available_columns_before_drop - specified_columns
            final_order_logs += [str(unspecified_columns)]
            final_order_logs += ['unspecified_columns']
            unspecified_columns = unspecified_columns - set(columns_to_be_dropped)
            final_order_logs += [str(unspecified_columns)]
            final_order_logs += ['final_columns_list']

            final_columns_list = actual_final_order + [col for col in list(unspecified_columns) if col in df.columns]
            final_order_logs += [str(final_columns_list)]
            LOG_FOLDER = 'AZLOGS'
            FINAL_ORDER_LOG_FILE = LOG_FOLDER + 'finalOrder.log'
            with open(FINAL_ORDER_LOG_FILE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(final_order_logs))
                logger.info(f'Final order log file created: {FINAL_ORDER_LOG_FILE}')
            logger.info(f'Final columns list: {final_columns_list}')
            try:
                # ! Subset df with the final columns list
                df = df[final_columns_list]

                # ! Proceed with saving the file
                df.to_excel(output_file, index=False)
                logger.info(f'Excel file created: {output_file}')
            except KeyError as e:
                logger.error(f"Error in applying final column order: {e}")
                # ! Handle the error or perform additional logging as needed

        df: pd.DataFrame = pd.read_excel(input_file)
        # ! Section 1
        columns_to_drop_due_to_concat: List[str] = []
        available_columns_before_drop = set(df.columns)
        df_total_rows: int = len(df)
        df_total_columns: int = len(df.columns)
        new_columns: Dict[str, Dict[str, Any]] = {
            'Qs': {
                'cols': ['requestQuery_s', 'originalRequestUriWithArgs_s', 'requestUri_s'],
                'include_names': True
            },
            'IPs': {
                'cols': ['clientIp_s', 'clientIP_s'],
                'include_names': False
            },
            'Host': {
                'cols': ['host_s', 'hostname_s', 'originalHost_s'],
                'include_names': False
            },
            'messagesD': {
                'cols': ['Message', 'action_s'],
                'include_names': True
            },
            'allD': {
                'cols': ['details_message_s', 'details_data_s', 'details_file_s', 'details_line_s'],
                'include_names': True
            },
            'rulesD': {
                'cols': ['engine_s', 'ruleSetVersion_s', 'ruleGroup_s', 'ruleId_s', 'ruleSetType_s', 'policyId_s', 'policyScope_s', 'policyScopeName_s'],
                'include_names': True
            },
            'serverD': {
                'cols': ['backendSettingName_s', 'noOfConnectionRequests_d', 'serverRouted_s', 'httpVersion_s'],
                'include_names': True
            },
            'Method_Type': {
                'cols': ['httpMethod_s', 'contentType_s'],
                'include_names': False
            },
            'TimeGenerated_eventD': {
                'cols': [
                    'eventTimestamp_s', 'errorMessage_s', 'errorCount_d', 'Environment_s', 'Region_s', 'ActivityName_s', 'operationalResult_s', 'ActivityId_g', 'ScaleUnit_s',
                    'NamespaceName_s'
                ],
                'include_names': True
            },
        }
        logger.info('[red]########## Final Excel Creation Started #########[/red]')
        logger.info(f'DF loaded from file: {input_file}')
        logger.info(f'Total Columns: {df_total_columns} - Total Rows: {df_total_rows}')
        # ! Section 2
        drop_and_create_columns(df, new_columns)
        # ! Section 3
        apply_final_column_order_and_save(df, final_column_order, output_file, available_columns_before_drop)





# class ExcelManager():

#     @staticmethod
#     def excelCreate(input_csv_file, output_excel_file, extraction_log_file, regex_patterns, string_patterns, col_to_val_patterns) -> None:
#         try:
#             df = pd.read_csv(input_csv_file, low_memory=False)

#             df = DataFrameManipulator.remove_patterns(dataframe=df,
#                                                       extraction_file=extraction_log_file,
#                                                       regex=regex_patterns,
#                                                       string=string_patterns,
#                                                       key_col_to_val_patts=col_to_val_patterns)

#             logger.info(f'Captured Logs: {extraction_log_file} - Rows: {df.shape[0]} - Col: {df.shape[1]}')

#             df.dropna(axis=1, how='all', inplace=True)  # ! Remove columns with all NaN values

#             if not df.empty:
#                 with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:  # pylint: disable=abstract-class-instantiated
#                     df.to_excel(writer, index=False, sheet_name='Sheet1')
#                     ExcelManager.format_excel_file(writer, df)
#                     logger.info(f'Excel formatted and created: {output_excel_file}')
#             else:
#                 logger.warning('DF is empty')
#         except Exception as e:
#             logger.error(f'E in create_excel: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

#     @staticmethod
#     def format_excel_file(writer, df) -> None:
#         """The `format_excel_file` function formats an Excel file by setting the font, font size, and column width for each column in a given DF."""
#         try:
#             workbook = writer.book
#             worksheet = writer.sheets['Sheet1']
#             cell_format = workbook.add_format({'font_name': 'Open Sans', 'font_size': 8})
#             worksheet.set_column('A:ZZ', None, cell_format)
#             for column in df:
#                 column_length = max(df[column].astype(str).apply(len).max(), len(column))
#                 col_idx = df.columns.get_loc(column)
#                 worksheet.set_column(col_idx, col_idx, column_length, cell_format)
#         except Exception as e:
#             logger.error(f'Error in format_excel_file: {e}', exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

#     @staticmethod
#     def createFinalExcel(input_file: str, output_file: str, extraction_log_file: str, columns_to_be_dropped: List[str], final_column_order: List[str],
#                          regex_patterns: Dict[str, str], string_patterns: Dict[str, str], col_to_val_patterns: Dict[str, str]) -> None:
#         """The `createFinalExcel` function takes an input file, applies various extraction patterns and
#         filters, and creates a final Excel file with specified column order and dropped columns."""

#         def concatenate_values(row: pd.Series, columns: List[str], include_col_names: bool = False) -> str:
#             """The function `concatenate_values` takes a row of a pandas DF, a list of column names, and an
#             optional flag to include column names, and returns a string concatenating the non-null values of the
#             specified columns."""
#             if include_col_names:
#                 return ', '.join([f'{col}: "{row[col]}"' for col in columns if pd.notna(row[col])])
#             return ', '.join([f'"{row[col]}"' for col in columns if pd.notna(row[col])])

#         def drop_and_create_columns(df: pd.DataFrame, new_columns: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
#             """The function `drop_and_create_columns` takes a DF and a dictionary of new columns as input,
#             creates new columns based on existing columns in the DF, drops columns that were used to
#             create the new columns, and returns the modified DF."""
#             logger.info('[red]########### columns_to_drop started ###########[/red]')
#             for col_name, info in new_columns.items():
#                 existing_cols = [col for col in info['cols'] if col in df.columns]
#                 missing_cols = [col for col in info['cols'] if col not in df.columns]
#                 if existing_cols:
#                     # ! Create new column by concatenating values from existing columns
#                     df[col_name] = df.apply(lambda row, cols=existing_cols: concatenate_values(row, cols, info['include_names']), axis=1)  # pylint: disable=W0640
#                     logger.info(f'New column "{col_name}" created with values from columns: {existing_cols} - Total rows: {df.shape[0]}')
#                     if col_name != 'Qs':
#                         columns_to_drop_due_to_concat.extend(existing_cols)
#                     else:
#                         logger.info(f'Columns kept: {existing_cols}')
#                 if missing_cols:
#                     logger.info(f'Columns not found in "{col_name}": {missing_cols}')
#             if columns_to_drop_due_to_concat:
#                 # ! Drop columns that were used to create new columns
#                 df.drop(columns=columns_to_drop_due_to_concat, inplace=True)
#                 dropped_columns = available_columns_before_drop - set(df.columns)
#                 logger.info(f'Columns dropped from concat: {columns_to_drop_due_to_concat}')
#                 if dropped_columns:
#                     logger.info("Columns dropped from concat:")
#                     for column in dropped_columns:
#                         logger.info(column)
#                 else:
#                     logger.info("No columns were dropped.")
#             logger.info('[red]########### columns_to_drop  from concat completed ###########[/red]')
#             logger.info(f'Columns dropped.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
#             logger.info(f'[green]Columns to be dropped started: {columns_to_be_dropped}[/green]')
#             available_columns_before_drop2 = set(df.columns)
#             cols_to_drop_from_func_def = [column for column in columns_to_be_dropped if column in df.columns]
#             # ! Drop columns specified in the function definition
#             if cols_to_drop_from_func_def:
#                 df.drop(columns=cols_to_drop_from_func_def, inplace=True)
#                 dropped_columns2 = available_columns_before_drop2 - set(df.columns)
#                 if dropped_columns2:
#                     logger.info("Columns dropped:")
#                     for column in dropped_columns2:
#                         logger.info(column)
#                 logger.info(f'Columns dropped 2.\nFinal DF: {df.shape[0]} rows {df.shape[1]} columns')
#                 logger.info('[green]Columns to be dropped ended: [/green]')
#             return df

#         def apply_final_column_order_and_save(df: pd.DataFrame, final_column_order: List[str], output_file: str, available_columns_before_drop: set) -> None:
#             """The function `apply_final_column_order_and_save` takes a DF, a list of column names, an
#             output file path, and a set of available columns as input, applies the specified column order to the
#             DF, saves the DF to an Excel file, and logs the process."""
#             final_order_logs = []
#             final_order_logs += ['Final order started']
#             final_order_logs += ['actual_final_order']
#             actual_final_order = [col for col in final_column_order if col in df.columns]
#             final_order_logs += [str(actual_final_order)]
#             final_order_logs += ['specified_columns']
#             specified_columns = set(actual_final_order)
#             final_order_logs += [str(specified_columns)]
#             final_order_logs += ['unspecified_columns']
#             unspecified_columns = available_columns_before_drop - specified_columns
#             final_order_logs += [str(unspecified_columns)]
#             final_order_logs += ['unspecified_columns']
#             unspecified_columns = unspecified_columns - set(columns_to_be_dropped)
#             final_order_logs += [str(unspecified_columns)]
#             final_order_logs += ['final_columns_list']

#             final_columns_list = actual_final_order + [col for col in list(unspecified_columns) if col in df.columns]
#             final_order_logs += [str(final_columns_list)]
#             FINAL_ORDER_LOG_FILE = LOG_FOLDER + 'finalOrder.log'
#             with open(FINAL_ORDER_LOG_FILE, 'w', encoding='utf-8') as f:
#                 f.write('\n'.join(final_order_logs))
#                 logger.info(f'Final order log file created: {FINAL_ORDER_LOG_FILE}')
#             logger.info(f'Final columns list: {final_columns_list}')
#             try:
#                 # ! Subset df with the final columns list
#                 df = df[final_columns_list]

#                 # ! Proceed with saving the file
#                 df.to_excel(output_file, index=False)
#                 logger.info(f'Excel file created: {output_file}')
#             except KeyError as e:
#                 logger.error(f"Error in applying final column order: {e}")
#                 # ! Handle the error or perform additional logging as needed

#         df: pd.DataFrame = pd.read_excel(input_file)
#         # ! Section 1
#         columns_to_drop_due_to_concat: List[str] = []
#         available_columns_before_drop = set(df.columns)
#         df_total_rows: int = len(df)
#         df_total_columns: int = len(df.columns)
#         new_columns: Dict[str, Dict[str, Any]] = {
#             'Qs': {
#                 'cols': ['requestQuery_s', 'originalRequestUriWithArgs_s', 'requestUri_s'],
#                 'include_names': True
#             },
#             'IPs': {
#                 'cols': ['clientIp_s', 'clientIP_s'],
#                 'include_names': False
#             },
#             'Host': {
#                 'cols': ['host_s', 'hostname_s', 'originalHost_s'],
#                 'include_names': False
#             },
#             'messagesD': {
#                 'cols': ['Message', 'action_s'],
#                 'include_names': True
#             },
#             'allD': {
#                 'cols': ['details_message_s', 'details_data_s', 'details_file_s', 'details_line_s'],
#                 'include_names': True
#             },
#             'rulesD': {
#                 'cols': ['engine_s', 'ruleSetVersion_s', 'ruleGroup_s', 'ruleId_s', 'ruleSetType_s', 'policyId_s', 'policyScope_s', 'policyScopeName_s'],
#                 'include_names': True
#             },
#             'serverD': {
#                 'cols': ['backendSettingName_s', 'noOfConnectionRequests_d', 'serverRouted_s', 'httpVersion_s'],
#                 'include_names': True
#             },
#             'Method_Type': {
#                 'cols': ['httpMethod_s', 'contentType_s'],
#                 'include_names': False
#             },
#             'TimeGenerated_eventD': {
#                 'cols': [
#                     'eventTimestamp_s', 'errorMessage_s', 'errorCount_d', 'Environment_s', 'Region_s', 'ActivityName_s', 'operationalResult_s', 'ActivityId_g', 'ScaleUnit_s',
#                     'NamespaceName_s'
#                 ],
#                 'include_names': True
#             },
#         }
#         logger.info('[red]########## Final Excel Creation Started #########[/red]')
#         logger.info(f'DF loaded from file: {input_file}')
#         logger.info(f'Total Columns: {df_total_columns} - Total Rows: {df_total_rows}')
#         df = DataFrameManipulator.remove_patterns(dataframe=df,
#                                                   extraction_file=extraction_log_file,
#                                                   regex=regex_patterns,
#                                                   string=string_patterns,
#                                                   key_col_to_val_patts=col_to_val_patterns)
#         logger.info('Manipulations.remove_patterns completed')
#         # ! Section 2
#         drop_and_create_columns(df, new_columns)
#         # ! Section 3
#         apply_final_column_order_and_save(df, final_column_order, output_file, available_columns_before_drop)

# class SystemUtils():

#     @staticmethod
#     def check_and_modify_permissions(path) -> None:
#         """Check and modify the permissions of the given path.
#         Args: path: The path to check and modify permissions for."""
#         try:
#             if not os.access(path, os.W_OK):
#                 logger.info(f"Write permission not enabled on {path}. Attempting to modify.")
#                 current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
#                 os.chmod(path, current_permissions | stat.S_IWUSR)
#                 logger.info(f"Write permission added to {path}.")
#             else:
#                 logger.info(f"Write permission is enabled on {path}.")
#         except Exception as e:
#             logger.info(f"Error modifying permissions: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
#             logger.error(f"Error modifying permissions for {path}: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)
