import os
import json
import librosa
import soundfile as sf
import random


def save_json_file(manifest_output_path, metadata):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(manifest_output_path), exist_ok=True)

    # Append the JSON metadata to the file
    with open(manifest_output_path, 'a') as json_file:  # Open in append mode
        json.dump(metadata, json_file)
        json_file.write('\n')  # Ensure each JSON object is on a new line

def apply_speed_perturbation(y, sr):
    speed_factor = random.choice([0.9, 1.1])
    y_perturbed = librosa.effects.time_stretch(y, rate=speed_factor)
    return y_perturbed


def augment_and_save(manifest_file, output_base_path, json_output_base_path, num_total_augmentations):
    # Open the file and read line by line
    with open(manifest_file, 'r') as file:
        for i, line in enumerate(file):

            parsed_line = json.loads(line.strip())

            original_path = parsed_line['audio_filepath']
            original_duration = parsed_line['duration']
            text = parsed_line['text']

            # Load the audio file
            y, sr = librosa.load(original_path, sr=None)

            # Extract the path components to match TRAIN/<recordname>/<speakername>
            relative_path = os.path.relpath(original_path, start='/mnt/c/Users/Robert/Downloads/timit-dataset/data')
            recordname, speakername, audiofile = relative_path.split(os.sep)[1:4]

            # Create the output directory based on the recordname and speakername
            output_dir = os.path.join(output_base_path, str(num_total_augmentations), recordname, speakername)
            os.makedirs(output_dir, exist_ok=True)

            # Save the original audio file in the output directory
            audio_output_path = os.path.join(output_dir, audiofile)
            sf.write(audio_output_path, y, sr)

            # Save the metadata for the original audio file
            original_metadata = {
                'audio_filepath': audio_output_path,
                'duration': original_duration,
                'text': text
            }

            manifest_output_path = os.path.join(json_output_base_path, str(num_total_augmentations), f"{recordname}_{speakername}_original.json")
            save_json_file(manifest_output_path, original_metadata)

            # Determine the number of augmentations to perform based on total augmentations
            num_total_augmentations_for_audio = 0
            if num_total_augmentations == 5 and i < 5:
                num_total_augmentations_for_audio = 1
            elif num_total_augmentations == 10:
                num_total_augmentations_for_audio = 1
            elif num_total_augmentations == 20:
                num_total_augmentations_for_audio = 2

            # Proceed with augmentation and saving logic here
            for j in range(num_total_augmentations_for_audio):
                augmented_audio = apply_speed_perturbation(y, sr)  # Placeholder for actual augmentation function
                output_file_path = os.path.join(output_dir, f"augmented_{j + 1}_{audiofile}")
                sf.write(output_file_path, augmented_audio, sr)

                # Save the metadata for the augmented file
                augmented_metadata = {
                    'audio_filepath': output_file_path,
                    'duration': librosa.get_duration(y=augmented_audio, sr=sr),
                    'text': text
                }

                save_json_file(manifest_output_path, augmented_metadata)


def process_all_manifests(manifests_folder, output_base_path, json_output_base_path, num_total_augmentations):
    # Iterate over all JSON files in the manifests folder
    for manifest_file in os.listdir(manifests_folder):
        print(manifest_file)
        if manifest_file.endswith('.json'):
            augment_and_save(os.path.join(manifests_folder, manifest_file), output_base_path, json_output_base_path, num_total_augmentations)


# Define paths and parameters
manifests_folder = "./timit-dataset/federated-manifests"
output_base_path = "./timit-dataset/augmented-audio"
json_output_base_path = "./tmp-noniid-augmented-manifests"
num_total_augmentations = [5, 10, 20]  # Number of total augmentaitons per manifest
# num_total_augmentations = 10 # Number of total augmentaitons per manifest
# num_total_augmentations = 10 # Number of total augmentaitons per manifest

for el in num_total_augmentations:
# Process all manifest files
    process_all_manifests(manifests_folder, output_base_path, json_output_base_path, el)
