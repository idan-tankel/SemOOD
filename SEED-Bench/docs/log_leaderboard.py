import wandb
import pandas as pd

# Read our CSV into a new DataFrame
new_iris_dataframe = pd.read_csv("SEED-Bench/leaderboard.csv")
iris_table = wandb.Table(dataframe=new_iris_dataframe)