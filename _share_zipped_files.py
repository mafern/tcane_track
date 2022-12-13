# Grab all of the prediction, blueprint and metadata files
# Zip them and save them for sharing

import glob
import zipfile
from datetime import date

MODEL_PATH = "saved_models/"
PREDICTIONS_PATH = "saved_predictions/"
SHARED_ZIP_FILES_PATH = "shared_zip_files/"

f_blueprint = glob.glob(MODEL_PATH + '/*/*_blueprint.json')
f_metadata = glob.glob(MODEL_PATH + '/*/*_metadata.json')
f_predictions = glob.glob(PREDICTIONS_PATH + '*_predictions.csv')
f_exp_settings = glob.glob('experiment_settings.py')
print(len(f_blueprint), len(f_metadata), len(f_predictions), len(f_exp_settings))

list_of_files = f_blueprint + f_metadata + f_predictions + f_exp_settings
print(len(list_of_files))

today = date.today()
save_zip_filename = SHARED_ZIP_FILES_PATH + 'hurricane_track_' + \
                    str(today.year) + str(today.month) + str(today.day) + '.zip'

with zipfile.ZipFile(save_zip_filename, 'w') as zipMe:
    for file in list_of_files:
        zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)
