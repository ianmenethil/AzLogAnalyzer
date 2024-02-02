import pandas as pd
import json
from openpyxl import Workbook
from typing import Literal
import os
import logging

logger = logging.getLogger(__name__)
OUTDIR = 'AZLOGS'


def compare_and_write_output(fp1, fp2) -> None:

    def read_headers(file_path) -> list[str]:
        # Convert WindowsPath to a string for compatibility
        file_path_str = str(file_path)

        if file_path_str.endswith('.csv'):
            df = pd.read_csv(file_path_str, low_memory=False)
            return list(df.columns)
        elif file_path_str.endswith('.xlsx'):
            df = pd.read_excel(file_path_str)
            return list(df.columns)
        elif file_path_str.endswith('.json'):
            with open(file_path_str, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list) and len(data) > 0:
                    return list(data[0].keys())
                elif isinstance(data, dict):
                    return list(data.keys())
                else:
                    raise ValueError("JSON file format not supported or empty.")
        else:
            raise ValueError("File format not supported. Please provide a CSV, XLSX, or JSON file.")

    def compare_headers(fp1, fp2) -> tuple[list[str], list[str], set[str], set[str]]:
        headers1 = read_headers(fp1)
        headers2 = read_headers(fp2)

        diff1 = set(headers1) - set(headers2)
        diff2 = set(headers2) - set(headers1)

        return headers1, headers2, diff1, diff2

    def write_output(file1_headers, file2_headers, diff1, diff2) -> None:
        wb = Workbook()
        ws = wb.active
        ws.title: Literal['Comparison']  # type: ignore
        ws.title = 'Comparison'
        # Assuming fp1 and fp2 are your file paths
        base_filename1 = os.path.basename(fp1)
        base_filename2 = os.path.basename(fp2)
        filename1 = os.path.splitext(base_filename1)[0]
        filename2 = os.path.splitext(base_filename2)[0]
        logger.info(f"File 1: {filename1} | File 2: {filename2}")
        out = filename1 + '_before' + filename2 + '.xlsx'
        output = f'{OUTDIR}/{out}'
        ws.append([f'{filename1}', f'{filename2}', 'Status'])
        # ws.append([f'{file1_ext} File 1', f'{file2_ext} File 2', 'Status'])
        all_headers = sorted(set(file1_headers) | set(file2_headers))
        for header in all_headers:
            if header in file1_headers and header not in file2_headers:
                ws.append([header, '', f'{header} Missing in File 2'])  # type: ignore
            elif header in file2_headers and header not in file1_headers:
                ws.append(['', header, f'{header} Missing in File 1'])  # type: ignore
            elif header in file1_headers and header in file2_headers:
                ws.append([header, header, 'Match'])  # type: ignore
        logger.info(f"Writing output to {output}")
        wb.save(output)

    output_file = f'{OUTDIR}/{fp1}_vs_{fp2}.xlsx'
    file1_headers, file2_headers, diff1, diff2 = compare_headers(fp1, fp2)
    logger.info(f"File 1 headers: {file1_headers}")
    logger.info(f"File 2 headers: {file2_headers}")
    logger.info(f"Missing headers in File 1: {diff1}")
    logger.info(f"Missing headers in File 2: {diff2}")
    write_output(file1_headers, file2_headers, diff1, diff2)
    logger.info(f"Output written to {output_file}")


# Setup the file paths
if __name__ == '__main__':
    # INPUTDIR = r'F:\_Azure\AzLogAnalyzer\AZLOGS\AZDIAG_TIMEBETWEEN\30-01-24'  # Choose your input directory
    # FILENAME1 = 'AZDIAG_TIMEBETWEEN_29-01-24_02.41PM.csv'  # Choose your first file name
    # FILENAME2 = 'AZDIAG_TIMEBETWEEN_FINAL_29-01-24_02.41PM.xlsx'  # Choose your second file name
    # FILEPATH1 = os.path.join(INPUTDIR, FILENAME1)
    # FILEPATH2 = os.path.join(INPUTDIR, FILENAME2)
    # OUTPUTFILE = os.path.join(f'{FILENAME1}_vs_{FILENAME2}.xlsx')  # Output file in the same directory
    compare_and_write_output(fp1=None, fp2=None)
