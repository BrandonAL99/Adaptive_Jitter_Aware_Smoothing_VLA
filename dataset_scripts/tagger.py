from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("BrandonAL/eval_my_smolvla_all_pickplace_part1", tag="_version_", repo_type="dataset")
print("Tag created successfully!")
