import os
from data_param import GROUPS, TASKS, RAW_DATA_FOLDER_PATH, SUBFOLDER

def delete_files_by_condition(path, condition):
    """
    Deletes files in the specified directory that meet the given condition.

    Args:
        path (str): Directory path to check for files.
        condition (func): A function that returns True if the file should be deleted.
    """
    for filename in os.listdir(path):
        if condition(filename):
            file_path = os.path.join(path, filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

def main(action):
    """
    Manages directory based on specified action to reset directory, remove Euler files,
    or remove duplicate files.

    Args:
        action (str): One of 'reset', 'remove_euler', or 'remove_duplicates' to specify the deletion action.
    """
    for group in GROUPS:
        for task in TASKS:
            path = os.path.join(RAW_DATA_FOLDER_PATH, group, SUBFOLDER, task)
            if os.path.exists(path):
                if action == 'reset':
                    delete_files_by_condition(path, lambda f: not f.endswith(".mat") and not f.endswith(".csv"))
                elif action == 'remove_euler':
                    delete_files_by_condition(path, lambda f: f.endswith("RPY.csv"))
                elif action == 'remove_duplicates':
                    delete_files_by_condition(path, lambda f: f.endswith("RPY_RPY.csv"))


if __name__ == '__main__':
    action = 'remove_euler'
    main(action)
