import pandas as pd


df=pd.read_csv('../data/df_eda.csv')

# We will only select rating and decoded_emojis columns as our features

df=df[['decoded_emojis','rating']]

# shift all rating encodings be 1 as -1 can't be read for Models

df['rating']=df['rating']+1

print(df['rating'].value_counts())

