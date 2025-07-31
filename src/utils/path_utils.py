import os, datetime

def get_output_dir(base_path: str, essential: dict, train_type: str = "dpo"):
    converted_model_id = slash_remove(essential["model_id"])
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dtype = str(essential["dtype"]).replace("torch.", "")
    save_dir = f"{current_time}_{converted_model_id}_r_{essential["r"]}_ra_{essential["lora_alpha"]}_rd_{essential["lora_dropout"]}_{dtype}_{train_type}"
    return os.path.join(base_path, save_dir)

def slash_remove(path: str):
    return path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
