import os
import subprocess
import traceback

batch_folder = 'batches'
script_file = 'ndvi.py'
log_file = 'batch_errors.log'

with open(log_file, 'w') as f:
    f.write("Batch Processing Error Log\n\n")

batch_files = [os.path.join(batch_folder, f) for f in sorted(os.listdir(batch_folder)) if f.endswith('.kml')]

for batch_file in batch_files:
    print(f"Processing batch file: {batch_file}")

    try:
        with open(script_file, 'r') as f:
            script_content = f.read()
        batch_file_path = os.path.abspath(batch_file).replace('\\', '/')
        modified_script = script_content.replace('PATH_TO_KML = \'lands/mach_test.kml\'',
                                                 f'PATH_TO_KML = \'{batch_file_path}\'')

        with open('temp_script.py', 'w') as f:
            f.write(modified_script)

        subprocess.run(['python', 'temp_script.py'], check=True)

        print(f"Finished processing: {batch_file}\n")

    except subprocess.CalledProcessError as e:
        with open(log_file, 'a') as f:
            f.write(f"Batch file failed: {batch_file}\n")
            f.write(f"Error: {str(e)}\n\n")

        print(f"Error occurred in batch file: {batch_file}. Logged the error.\n")

    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Unexpected error with batch file: {batch_file}\n")
            f.write(f"Error: {traceback.format_exc()}\n\n")

        print(f"Unexpected error occurred in batch file: {batch_file}. Logged the error.\n")

os.remove('temp_script.py')

print("All batch files processed.")

