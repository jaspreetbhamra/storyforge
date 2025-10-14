import wandb

wandb.init(project="test-project", entity="jay_b")

wandb.log({"test": 1})
wandb.finish()

"""
Make sure you've run the following in case you're seeing errors:
pip install wandb
wandb login
wandb online
wandb enabled
"""


# import os

# import wandb

# # Try to login with your key directly
# api_key = "<KEY>"  # Paste your actual key
# os.environ["WANDB_API_KEY"] = api_key

# print(f"API Key length: {len(api_key)}")
# print(f"First 10 chars: {api_key[:10]}")
# print(f"Last 10 chars: {api_key[-10:]}")

# try:
#     wandb.login(key=api_key, relogin=True)
#     print("✅ Login successful!")

#     run = wandb.init(project="test-project", mode="online")
#     print("✅ Init successful!")
#     wandb.finish()
# except Exception as e:
#     print(f"❌ Error: {e}")
