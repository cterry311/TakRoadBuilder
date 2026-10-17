from tak_engine import Tak
from model.tak_net import TakNet
import sqlite3
import torch
import numpy as np

conn = sqlite3.connect("games.db")
cursor = conn.cursor()

# fetch data, split it into training, testing, validation, split it by game rather than by position to avoid leaking data between training and testing sets

cursor.execute('''
    SELECT game_id
    FROM training_positions
    GROUP BY game_id
''')

EPOCHS = 1000

for i in range(EPOCHS):
    pass




cursor.close()
conn.close()