import pandas as pd
import json
from openpyxl import Workbook
import os


def compare_and_write_output(fp1, fp2, output_file) -> None:

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

    def write_output(file1_headers, file2_headers, diff1, diff2, output_file) -> None:
        wb = Workbook()
        ws = wb.active
        ws.title = 'Comparison'
        # Strip file extensions from the file names
        file1_ext = os.path.splitext(os.path.basename(fp1))[0]
        file2_ext = os.path.splitext(os.path.basename(fp2))[0]
        base_filename1 = os.path.basename(fp1)
        base_filename2 = os.path.basename(fp2)
        # Adjust the header names according to your file names or preferences
        ws.append([f'{base_filename1}{file1_ext} File 1', f'{base_filename2}{file2_ext} File 2', 'Status'])
        # ws.append([f'{file1_ext} File 1', f'{file2_ext} File 2', 'Status'])

        all_headers = sorted(set(file1_headers) | set(file2_headers))
        for header in all_headers:
            if header in file1_headers and header not in file2_headers:
                ws.append([header, '', f'{header} Missing in File 2'])
            elif header in file2_headers and header not in file1_headers:
                ws.append(['', header, f'{header} Missing in File 1'])
            elif header in file1_headers and header in file2_headers:
                ws.append([header, header, 'Match'])

        wb.save(output_file)

    file1_headers, file2_headers, diff1, diff2 = compare_headers(fp1, fp2)
    write_output(file1_headers, file2_headers, diff1, diff2, output_file)
    print(f"Output written to {output_file}")


# Setup the file paths
if __name__ == '__main__':
    INPUTDIR = r'F:\_Azure\AzLogAnalyzer\AZLOGS\AZDIAG_TIMEBETWEEN\30-01-24'  # Choose your input directory
    FILENAME1 = 'AZDIAG_TIMEBETWEEN_29-01-24_02.41PM.csv'  # Choose your first file name
    FILENAME2 = 'AZDIAG_TIMEBETWEEN_FINAL_29-01-24_02.41PM.xlsx'  # Choose your second file name
    FILEPATH1 = os.path.join(INPUTDIR, FILENAME1)
    FILEPATH2 = os.path.join(INPUTDIR, FILENAME2)
    OUTPUTFILE = os.path.join(f'{FILENAME1}_vs_{FILENAME2}.xlsx')  # Output file in the same directory
    compare_and_write_output(FILEPATH1, FILEPATH2, OUTPUTFILE)
