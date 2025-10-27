import os
import requests
import pandas as pd

def cv_decode(root_dir: str, csv_file_path: str):
    """
    Calls the ASR API to transcribe the audio files in the root directory and save the results to the csv file
    
    Args:
        root_dir (str): The root directory containing the audio files (e.g. the cv-valid-dev folder)
        csv_file_path (str): The csv file to save the results (e.g. the cv-valid-dev.csv file)
    """
    df = pd.read_csv(csv_file_path)
    
    generated_texts = []
    
    for filename in df['filename']:
        try:
            with open(root_dir + '/' + filename, 'rb') as f:
                response = requests.post(
                    'http://localhost:8001/asr', 
                    files={'file': f})
            
            if response.status_code == 200:
                result = response.json()
                generated_texts.append(result.get('transcription', ''))
                print(f"Processed {filename}: {result.get('transcription', '')}")
            else:
                generated_texts.append('')
                print(f"Error processing {filename}: {response.status_code}")
        except Exception as e:
            generated_texts.append('')
            print(f"Error processing {filename}: {str(e)}")
    
    df['generated_text'] = generated_texts
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))

    root_dir = os.path.join(script_dir, '..', 'data', 'common_voice', 'cv-valid-dev')

    csv_file_path = os.path.join(script_dir, 'cv-valid-dev.csv')

    cv_decode(root_dir=root_dir, csv_file_path=csv_file_path)