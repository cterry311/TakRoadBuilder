import socket, threading, time
from engine_clients.roadbuilder import RoadBuilder
import tak_engine
import playtak_connection.parsing as parsing
import os
from dotenv import load_dotenv

load_dotenv()

HOST, PORT = "playtak.com", 10000
sock = socket.create_connection((HOST, PORT))
f = sock.makefile("rw", encoding="utf-8", newline="\n")
lock = threading.Lock()

def send(line):
    with lock:
        print(">", line)
        f.write(line + "\n")
        f.flush()

def pinger():
    while True:
        time.sleep(30)
        send("PING")


class PlayTakGame:

    def __init__(self, game_no, my_color, total_time, time_increment, half_komi):
        # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?>
        self.game = tak_engine.Tak(komi=half_komi)
        self.time_increment = time_increment
        self.total_time = total_time
        self.color = my_color
        self.game_no = game_no
        if my_color == "white":
            self.send_move()

    def send_move(self):
        pass





threading.Thread(target=pinger, daemon=True).start()

game_no = None
my_color = None

engine = RoadBuilder('model_checkpoints/version_9_34.pt')
game = tak_engine.Tak(komi=2)
sent_accsept = False

for raw in f:                       # iterates line by line; ends at EOF
    line = raw.rstrip("\n")
    print("<", line)
    words = line.split(" ")

    if line.startswith("Login"):
        send("Client MyBot")
        send(f"Login {os.getenv('PLAYTAK_USERNAME')} {os.getenv('PASSWORD')}")         # or: send(f"Login {user} {password}")
    elif line.startswith("Welcome "):       # note the trailing space
        send("Seek 6 900 10 A 4 30 1 1 0 ") # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?> # note, trailing space is important!!!
    elif line.startswith("Online") and not sent_accsept:
        pass
        # send(f"Accept {112}")
        # sent_accsept = True
    elif line.startswith("Game Start"):
        game_no = words[2]
        my_color = words[7]
        print("Game started, I am", my_color)
        if my_color == "white":
            send(f"Game#{game_no} P A1")    # your first move
            game.make_move("a1")
    elif game_no and line.startswith(f"Game#{game_no} "):
        rest = line[len(f"Game#{game_no} "):]
        if rest.startswith(("P ", "M ")):
            print("Opponent moved:", rest)
            ptn_move = parsing.server_to_ptn(rest)
            game.make_move(ptn_move)
            best_move = engine.best_move(game.to_tps(), time_limit=18000)[0]
            game.make_move(best_move)
            server_move = parsing.ptn_to_server(best_move)
            send(f"Game#{game_no} {server_move}")
            # decide your move here, then send(f"Game#{game_no} P C3") etc.
        elif rest.startswith(("Over", "Abandoned")):
            print("Game over:", rest)
            break

# < Seek new 217 RoadBuilderBot    6    900   10    A   4   30   1   1   0   0    0    0
# < Seek new 220 RoadBuilderBot    6    900   10    A   4   30   1   1   0   0    0    0
# < Seek new 112 Tiltak_Bot        6    600   20    A   4   30   1   0   0   35   1200



# output for Seek 6 900 10 A 4 30 1 1 0   no trailing space

# < Welcome!
# < Login or Register
# > Client MyBot
# > Login RoadBuilderBot capstone
# < OK
# < Welcome RoadBuilderBot!
# > Seek 6 900 10 A 4 30 1 1 0
# < Seek new 192 NoviceBot5 5 420 1 A 0 21 1 0 0 0 0
# < Seek new 231 BeginnerBot5 5 420 1 A 0 21 1 0 0 0 0
# < Seek new 169 SlateBot 6 600 20 A 4 30 1 0 0 35 600
# < Seek new 207 ElirBot 5 600 30 A 0 21 1 0 0 0 0
# < Seek new 112 Tiltak_Bot 6 600 20 A 4 30 1 0 0 35 1200
# < Seek new 209 GravelBot 5 900 30 A 0 21 1 0 0 0 0
# < Seek new 83 TakticianBot 6 1200 30 A 0 30 1 0 0 0 0
# < Seek new 211 Guest210 6 1200 20 W 7 30 1 0 0 0 0
# < Seek new 212 EltheBot 5 60 5 A 0 21 1 0 0 0 0
# < Seek new 21 IntermediateBot6 6 600 1 A 0 30 1 0 0 0 0
# < Seek new 22 NoviceBot6 6 600 1 A 0 30 1 0 0 0 0
# < Seek new 54 IntermediateBot5 5 420 1 A 0 21 1 0 0 0 0
# < Seek new 24 AdvancedBot5 5 420 1 A 0 21 1 0 0 0 0
# < Seek new 184 AdvancedBot6 6 600 1 A 0 30 1 0 0 0 0
# < Seek new 216 CobbleBot 5 900 30 A 0 21 1 0 0 0 0
# < Seek new 218 BeginnerBot6 6 600 1 A 0 30 1 0 0 0 0
# < Seek new 219 Guest222 6 1200 20 A 4 30 1 0 0 0 0
# < Seek new 221 FriendlyBot 6 900 30 A 0 30 1 0 0 0 0
# < Seek new 158 RelarBot 6 900 30 A 0 30 1 0 0 0 0
# < Online 30
# < OnlinePlayers ["phenomenoire", "ShrewdConjurer", "SwedishChef", "Guest210", "Guest222", "pure_chance", "Guest227", "Guest233", "Guest175", "Guest231", "Guest225", "RoadBuilderBot"]
# < Seek new 232 RoadBuilderBot 6 900 10 A 0 30 1 0 0 0 0
# < Online 29
# < OnlinePlayers ["phenomenoire", "ShrewdConjurer", "SwedishChef", "Guest210", "Guest222", "pure_chance", "Guest227", "Guest233", "Guest175", "Guest231", "RoadBuilderBot"]
# < Online 30
# < OnlinePlayers ["phenomenoire", "ShrewdConjurer", "SwedishChef", "Guest210", "Guest222", "pure_chance", "Guest227", "Guest233", "Guest225", "Guest175", "Guest231", "RoadBuilderBot"]
# < Seek remove 232 RoadBuilderBot 6 900 10 A 0 30 1 0 0 0 0
# < GameList Add 881153 RoadBuilderBot Guest225 6 900 10 0 30 1 0 0 0 0
# < Game Start 881153 6 RoadBuilderBot vs Guest225 white 900 0 30 1 0 0
# Game started, I am white
# > Game#881153 P A1
# < Game#881153 Time 910 900
# < OK
# < Game#881153 Over 1-0
# Game over: Over 1-0