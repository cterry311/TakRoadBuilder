import socket, threading, time
from engine_clients.roadbuilder import RoadBuilderGame
import tak_engine
import playtak_connection.parsing as parsing
import os
from dotenv import load_dotenv
import queue
from abc import ABC, abstractmethod


load_dotenv()

HOST, PORT = "playtak.com", 10000
sock = socket.create_connection((HOST, PORT))
f_write = sock.makefile("rw", encoding="utf-8", newline="\n")
lock = threading.Lock()
lines = queue.Queue()

def reader():
    f_read = sock.makefile("r", encoding="utf-8", newline="\n")
    while True:
        line = f_read.readline()
        if not line:
            break
        lines.put(line.strip())


def send(line):
    with lock:
        print(">", line)
        f_write.write(line + "\n")
        f_write.flush()

def pinger():
    while True:
        time.sleep(30)
        send("PING")


threading.Thread(target=pinger, daemon=True).start()
threading.Thread(target=reader, daemon=True).start()


class EventHandler(ABC):
    @abstractmethod
    def handle_event(self, line: str):
        print("<", line)

    @abstractmethod
    def run_idle(self) -> EventHandler:
        pass

class LoginEventHandler(EventHandler):
    def __init__(self, user, password, future_handler: EventHandler):
        self.user = user
        self.password = password
        self.logged_in = False
        self.future_handler = future_handler

    def handle_event(self, line):
        super().handle_event(line)
        if line.startswith("Login or Register"):
            send(f"Login {self.user} {self.password}")
        elif line.startswith(f"Welcome {self.user}"):
            self.logged_in = True

    def run_idle(self):
        if self.logged_in:
            return self.future_handler
        else:
            return self

class SeekEventHandler(EventHandler):
    def __init__(self, start_time, increment, komi, color, unrated):
        self.start_time = start_time
        self.increment = increment
        self.komi = komi
        self.color = color
        self.unrated = unrated
        self.game_no = None
        self.game_color = None
        self.sent_seek = False

    def clone(self):
        return SeekEventHandler(self.start_time, self.increment, self.komi, self.color, self.unrated)

    def handle_event(self, line):
        super().handle_event(line)
        if line.startswith("Game Start"):
            self.game_no = int(line.split(" ")[2])
            self.game_color = line.split(" ")[7]

    def run_idle(self):
        if self.game_no is None:
            if not self.sent_seek:
                send(f"Seek {6} {self.start_time} {self.increment} {self.color} {int(self.komi * 2)} {30} {1} {self.unrated} {0} ")  # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?> # note, trailing space is important!!!
                self.sent_seek = True
            return self
        else:
            return GameEventHandler(self.game_no, self.game_color, self.start_time, self.increment, self.komi, future_handler=self.clone())


class AcceptEventHandler(EventHandler):
    def __init__(self, challenger_name):
        self.challenger_name = challenger_name
        self.sent_accept = False
        self.game_no = None
        self.game_color = None
        self.game_settings = {}

    def handle_event(self, line):
        super().handle_event(line)
        if line.startswith("Seek new "):
            name = line.split(" ")[3]
            if name == self.challenger_name:
                send(f"Accept {int(line.split(' ')[2])}")
                self.sent_accept = True
                self.game_settings = {
                    'total_time': int(line.split(' ')[5]),
                    'time_increment': int(line.split(' ')[6]),
                    'komi': int(int(line.split(' ')[8]) // 2),
                }
        if line.startswith("Game Start"):
            self.game_no = int(line.split(" ")[2])
            self.game_color = line.split(" ")[7]

    def clone(self):
        return AcceptEventHandler(self.challenger_name)

    def run_idle(self):
        if self.game_no is None:
            return self
        else:
            return GameEventHandler(self.game_no, self.game_color, **self.game_settings, future_handler=self.clone())




class GameEventHandler(EventHandler):

    def __init__(self, game_no, my_color, total_time, time_increment, komi, future_handler: EventHandler = None):
        # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?>
        self.game = tak_engine.Tak(komi=komi)
        self.time_increment = time_increment
        self.current_time = total_time
        self.color = my_color
        self.game_no = game_no
        self.engine = RoadBuilderGame('model_checkpoints/version_9_34.pt')
        self.current_q: float| None = None
        self.game_over = False
        self.future_handler = future_handler
        if my_color == "white":
            self.send_move()


    def send_move(self):
        time_to_move = self.time_increment
        if self.current_q is None:
            pass
        elif self.current_q < -0.9:
            time_to_move = self.time_increment * 2
        elif self.current_q < -0.7:
            time_to_move = self.time_increment * 2.5
        elif self.current_q < -0.5:
            time_to_move = self.time_increment * 3
        elif self.current_q < -0.3:
            time_to_move = self.time_increment * 4
        elif self.current_q < -0.1:
            time_to_move = self.time_increment * 3
        elif self.current_q < 0.1:
            time_to_move = self.time_increment * 2
        elif self.current_q < 0.3:
            time_to_move = self.time_increment * 1.5
        elif self.current_q < 0.5:
            time_to_move = self.time_increment * 0.8
        elif self.current_q < 0.8:
            time_to_move = self.time_increment * 0.5

        if self.current_time - time_to_move <= 0:
            time_to_move = self.current_time - 5
            if time_to_move <= 0:
                time_to_move = self.current_time - 2
            if time_to_move <= 0:
                time_to_move = self.current_time - 1
            if time_to_move <= 0:
                time_to_move = 0.2

        best_move, _, _, q = self.engine.best_move(time_limit=int(time_to_move * 1000))
        self.engine.make_move(best_move)
        self.current_q = q
        self.game.make_move(best_move)
        server_move = parsing.ptn_to_server(best_move)
        send(f"Game#{self.game_no} {server_move}")

    def handle_event(self, line: str):
        super().handle_event(line)
        if line.startswith(f"Game#{self.game_no}"):
            sent_type = line.split(" ")[1]
            rest = line[len(f"Game#{self.game_no} "):]
            if sent_type in ['P', 'M']:
                ptn_move = parsing.server_to_ptn(rest)
                self.game.make_move(ptn_move)
                self.engine.make_move(ptn_move)
                self.send_move()
            elif sent_type == "Time":
                self.current_time = int(line.split(" ")[2])
            elif sent_type == "Over":
                self.game_over = True


    def run_idle(self):
        if self.game_over:
            return self.future_handler
        else:
            self.engine.bake_tree(time_limit=500)
            return self

seek_action = SeekEventHandler(600, 20, 2, "A", 0)
handler: EventHandler | None = LoginEventHandler(os.getenv('PLAYTAK_USERNAME'), os.getenv('PASSWORD'), seek_action.clone())
while True:

    handler = handler.run_idle()

    if handler is None:
        break

    while True:
        try:
            line = lines.get_nowait()
            handler.handle_event(line)
        except queue.Empty:
            break





'''
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

'''