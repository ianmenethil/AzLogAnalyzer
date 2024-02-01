import logging
import pandas as pd

logger = logging.getLogger(__name__)


class LogManager():

    @staticmethod
    def save_removed_rows_to_raw_logs(DF_removed_rows, filename) -> None:
        try:
            removed_count = len(DF_removed_rows)
            print(f"Removed count: {removed_count}")
            log_data = [list(row.values) for _, row in DF_removed_rows.iterrows()]
            df = pd.DataFrame(log_data, columns=DF_removed_rows.columns)
            df.to_csv(filename, index=False)
        except Exception as e:
            logger.error(f'Error in save_removed_rows_to_raw_logs: {e}', exc_info=True, stack_info=True)

    # @staticmethod
    # def save_raw_logs(data: str, filename: str) -> None:
    #     try:
    #         logger.info(f"Saving raw logs to file: {filename}")
    #         calling_function = None
    #         frame = inspect.currentframe()
    #         if frame and frame.f_back:
    #             calling_function = frame.f_back.f_code.co_name
    #             logger.info(f"Calling function: {calling_function}")
    #         else:
    #             logger.warning("No calling function found.")
    #         with open(filename, "w", encoding="utf-8") as file:
    #             file.write(f"Calling function: {calling_function}\n\n{data}")
    #     except IOError as e:
    #         logger.error(f"Error saving raw logs: {e}", exc_info=True, stack_info=True, extra={'color': 'red'}, stacklevel=2)

    # @staticmethod
    # def save_removed_rows_to_raw_logs(removed_rows, column, pattern, filename) -> None:
    #     try:
    #         removed_count = len(removed_rows)
    #         log_data = f"Column: '{column}', Pattern: '{pattern}', Removed Rows Count: {removed_count}\n"
    #         LF()
    #         for _, row in removed_rows.iterrows():
    #             log_data += f"{row[column]}"
    #             additional_columns = [KEYSEARCH_requestQuery_s, KEYSEARCH_originalRequestUriWithArgs_s, KEYSEARCH_requestUri_s]
    #             if additional_info := [f"{col}: {row[col]}" for col in additional_columns if col in row]:
    #                 log_data += ", ".join(additional_info) + "\n"
    #         FileHandler.save_raw_logs(log_data, filename)
    #     except Exception as e:
    #         logger.error(f'Error in save_removed_rows_to_raw_logs: {e}', exc_info=True, stack_info=True)
    # @staticmethod
    # def save_removed_rows_to_raw_logs(DF_removed_rows, filename) -> None:
    #     try:
    #         removed_count = len(DF_removed_rows)
    #         print(f"Removed count: {removed_count}")
    #         log_data = [list(row.values) for _, row in DF_removed_rows.iterrows()]
    #         df = pd.DataFrame(log_data, columns=DF_removed_rows.columns)
    #         df.to_csv(filename, index=False)
    #     except Exception as e:
    #         logger.error(f'Error in save_removed_rows_to_raw_logs: {e}', exc_info=True, stack_info=True)
