import os

agents_dir = os.path.join("./data", "llamaindex_docs", "agents")
for folder_name in os.listdir(agents_dir):
    if os.path.isdir(os.path.join(agents_dir, folder_name)):
        print(folder_name)
        # move the file in the agents_dir with the same name + "_summary.pkl" to the folder
        os.rename(
            os.path.join(agents_dir, folder_name + "_summary.pkl"),
            os.path.join(agents_dir, folder_name, folder_name + "_summary.pkl"),
        )
