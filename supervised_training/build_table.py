import sqlite3
import numpy as np

conn = sqlite3.connect("games.db")
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT game_id FROM training_positions ORDER BY game_id")
game_ids = [row[0] for row in cursor.fetchall()]   # plain Python ints, sorted

rng = np.random.default_rng(42)
order = rng.permutation(len(game_ids))
game_ids = [game_ids[i] for i in order]

n = len(game_ids)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

splits = (
    [(g, 0) for g in game_ids[:n_train]] +                 # 0 = train
    [(g, 1) for g in game_ids[n_train:n_train + n_val]] +  # 1 = val
    [(g, 2) for g in game_ids[n_train + n_val:]]           # 2 = test
)

cursor.execute("DROP TABLE IF EXISTS game_splits")
cursor.execute("CREATE TABLE game_splits (game_id INTEGER PRIMARY KEY, split TINYINT)")
cursor.executemany("INSERT INTO game_splits VALUES (?, ?)", splits)   # executemany is fine here
conn.commit()

# 0 is training, 1 is validation, 2 is testing
def fetch_split(split):
    cursor.execute('''
        SELECT p.game_result, p.tps, p.move_raw
        FROM training_positions p
        JOIN game_splits s ON s.game_id = p.game_id
        WHERE s.split = ?
    ''', (split,))
    return cursor.fetchall()

train_positions = fetch_split(0)
val_positions = fetch_split(1)
test_positions = fetch_split(2)
