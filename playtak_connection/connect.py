"""
connect to playtak.com as a bot account
uses the raw model files, not the onnx releases
"""
import socket, threading, time
from engine_clients.roadbuilder import RoadBuilderGame, RoadBuilder
import tak_engine
import playtak_connection.parsing as parsing
import os
from dotenv import load_dotenv
import queue
from abc import ABC, abstractmethod
import json


load_dotenv()

HOST, PORT = "playtak.com", 10000
sock = socket.create_connection((HOST, PORT))
f_write = sock.makefile("w", encoding="utf-8", newline="\n")
f_read = sock.makefile("r", encoding="utf-8", newline="\n")
lock = threading.Lock()
lines = queue.Queue()

def reader():
    """
    read lines from the socket and put them in the queue
    :return:
    """
    while True:
        line = f_read.readline()
        if not line:
            break
        lines.put(line.strip())


def send(line: str):
    """
    write a line to the socket
    :param line: the line to write
    :return:
    """
    with lock:
        print(">", line)
        f_write.write(line + "\n")
        f_write.flush()

def pinger():
    """
    ping the socket every 30 seconds
    :return:
    """
    while True:
        time.sleep(30)
        send("PING")


threading.Thread(target=pinger, daemon=True).start()
threading.Thread(target=reader, daemon=True).start()


class EventHandler(ABC):
    """
    abstact class for connecting and interacting with the server
    """
    @abstractmethod
    def handle_event(self, line: str) -> EventHandler:
        """
        preform an action depending on the line recieved from the server

        :param line: the line recieved from the server
        :return: a instance of the EventHandler class, to replace the current handler, return self to not replace
        """
        print("<", line)
        return self


    @abstractmethod
    def run_idle(self) -> EventHandler:
        """
        run an idle action, this is called when there are no lines to handle
        :return: the event handler to replace this handler, return self to not replace
        """
        pass

class LoginEventHandler(EventHandler):
    """
    the handler when initially connecting, handles the login process
    """
    def __init__(self, user: str, password: str, future_handler: EventHandler):
        """

        :param user: the username to login with
        :param password: the password to login with
        :param future_handler: the event handler to replace this handler once the login is complete
        """
        self.user = user
        self.password = password
        self.logged_in = False
        self.future_handler = future_handler

    def handle_event(self, line):
        """
        login when requested, once logged in, send the future handler

        :param line: the line recieved from the server
        :return: self or future handler
        """
        super().handle_event(line)
        if line.startswith("Login or Register"):
            send(f"Login {self.user} {self.password}")
        elif line.startswith(f"Welcome {self.user}"):
            self.logged_in = True
            return self.future_handler
        return self

    def run_idle(self):
        """
        do nothing, if logged in, return the future handler, otherwise return self

        :return: self or future handler
        """
        if self.logged_in:
            return self.future_handler
        else:
            return self

class SeekEventHandler(EventHandler):
    """
    event handler for seeking a game
    """
    def __init__(self, start_time: int, increment: int, komi: int, color: str, unrated: int):
        """

        :param start_time: the start time for the game clock in seconds
        :param increment: the time increment for the game clock in seconds
        :param komi: the komi for the game
        :param color: the color, 'A' for any color, 'W' for white, 'B' for black
        :param unrated: 0 for rated, 1 for unrated
        """
        self.start_time = start_time
        self.increment = increment
        self.komi = komi
        self.color = color
        self.unrated = unrated
        self.game_no = None
        self.game_color = None
        self.sent_seek = False
        self.opponent_name = None

    def clone(self) -> SeekEventHandler:
        """
        create a duplicate of itself, with its own parameters
        :return: a copy of itself without internal state
        """
        return SeekEventHandler(self.start_time, self.increment, self.komi, self.color, self.unrated)

    def handle_event(self, line):
        """
        if recieved the Game Start event, set internal state to accsepted
        :param line: the line recieved from the server
        :return: self
        """
        super().handle_event(line)
        if line.startswith("Game Start"):
            self.game_no = int(line.split(" ")[2])
            self.game_color = line.split(" ")[7]
            if line.split(" ")[4].lower() != 'roadbuilderbot':
                self.opponent_name = line.split(" ")[4]
            else:
                self.opponent_name = line.split(" ")[6]
        return self

    def run_idle(self):
        """
        if game has started, return a game even handler, otherwise return self
        :return: self, or GameEventHandler
        """
        if self.game_no is None:
            if not self.sent_seek:
                send(f"Seek {6} {self.start_time} {self.increment} {self.color} {int(self.komi * 2)} {30} {1} {self.unrated} {0} ")  # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?> # note, trailing space is important!!!
                self.sent_seek = True
            return self
        else:
            return GameEventHandler(self.game_no, self.game_color, self.start_time, self.increment, self.komi, future_handler=self.clone(), opponent_name=self.opponent_name)


class AcceptEventHandler(EventHandler):
    """
    accept a game sought by somebody else
    """
    def __init__(self, challenger_name: str):
        """

        :param challenger_name: the username of the challenger to seek
        """
        self.challenger_name = challenger_name
        self.sent_accept = False
        self.game_no = None
        self.game_color = None
        self.game_settings = {}

    def handle_event(self, line):
        """
        if the line is a seek request from the challenger, accept it
        :param line: the line recieved from the server
        :return: self
        """
        super().handle_event(line)
        if line.startswith("Seek new "):
            print(line)
            name = line.split(" ")[3]
            print(name)
            if name == self.challenger_name:
                print("Accepting")
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
        return self

    def clone(self) -> AcceptEventHandler:
        """
        create a duplicate of itself, with its own parameters
        :return: AcceptEventHandler
        """
        return AcceptEventHandler(self.challenger_name)

    def run_idle(self):
        """
        if game was accepted, return a game even handler, otherwise return self
        :return: self or GameEventHandler
        """
        if self.game_no is None:
            return self
        else:
            return GameEventHandler(self.game_no, self.game_color, **self.game_settings, future_handler=self.clone(), opponent_name=self.challenger_name)


# c_puct
# urgency
# alpha
# eps

# max_sims

# temperature

"""
Difficutly settings for the bot
"""
DIFFICULTY_SETTINGS = {
    'beginner': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.1,
            "eps": 0.3,
            "max_sims": 1_000,
            "temperature": 1.0,
            "difficulty": 'beginner'
    },
    'easy': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.05,
            "eps": 0.25,
            "max_sims": 1_500,
            "temperature": 0.85,
            "difficulty": 'medium'
    },
    'medium': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.01,
            "eps": 0.25,
            "max_sims": 2_000,
            "temperature": 0.7,
            "difficulty": 'medium'
    },
    'hard': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.01,
            "eps": 0.5,
            "max_sims": 6_000,
            "temperature": 0.4,
            "difficulty": 'hard'
    },
    'expert': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.01,
            "eps": 0.5,
            "max_sims": 15_000,
            "temperature": 0.1,
            "difficulty": 'expert'
    },
    'master': {
            "c_puct": 1.5,
            "urgency": -0.2,
            "alpha": 0.01,
            "eps": 0.5,
            "max_sims": 30_000,
            "temperature": 0.05,
            "difficulty": 'expert'
    },
    'max': {
            "c_puct": 1.5,         # how much the inital prior should be weighed when eval is availible
            "urgency": -0.2,       # how much the engine should be encoraged to try unexplored moves
            "alpha": 0.1,          # how spread out the noise is, lower values will give spiky noise
            "eps": 0.25,           # how much of the value is noise
            "max_sims": 0,         # the maximum number of sims allowed 0 for unlimited sims
            "temperature": 0,      # the how flat the distribution for sampling the move is, 0 is argmax,
            "difficulty": "max"
    },
}

class GameEventHandler(EventHandler):
    """
    event handler for playing a game
    """
    def __init__(self, game_no: int, my_color: str, total_time: int, time_increment: int, komi: int, future_handler: EventHandler = None, opponent_name: str = None):
        """
        set up the game, send message to opponent describing how to use chat commands, make the first move if white

        :param game_no: the game number
        :param my_color: my own color, either 'white' or 'black'
        :param total_time: the total time on the game clock in seconds
        :param time_increment: the time increment per move in seconds
        :param komi: komi for the game
        :param future_handler: the event handler to return when the game is over
        :param opponent_name: opponents name
        """
        # Seek <size> <time> <increment> <color> <half_komi> <flats> <capstones> <unrated> <tournament?>
        self.game = tak_engine.Tak(komi=komi)
        self.time_increment = time_increment
        self.current_time = total_time
        self.color = my_color
        self.game_no = game_no

        game_config = load_settings()

        self.c_puct = game_config['c_puct']
        self.urgency = game_config['urgency']
        self.max_sims = game_config['max_sims']
        self.alpha = game_config['alpha']
        self.eps = game_config['eps']
        self.temperature = game_config['temperature']
        self.difficulty = game_config['difficulty']

        self.move_records = []

        self.engine = RoadBuilderGame('model_checkpoints/version_14_36.pt',
                                      capacity=200_000_000,
                                      c_puct=self.c_puct,
                                      urgency=self.urgency,
                                      alpha=self.alpha,
                                      eps=self.eps)
        self.current_q: float| None = None
        self.game_over = False
        self.future_handler = future_handler
        if opponent_name is not None:
            send(f"Tell {opponent_name} roadbuilder difficulty is {self.difficulty}")
            send(f"Tell {opponent_name} how to use https://github.com/cterry311/TakRoadBuilder/blob/master/RoadBuilderBot_settings.md")

        if my_color == "white":
            self.send_move()


    def send_move(self):
        """
        send a move to the server, make the move in the engine
        :return:
        """
        if self.game.result() != 0:
            return

        if self.current_q is None:
            q_modifier = 1
        else:
            q_modifier = (-0.8 * self.current_q) + 1

        time_modifier = (self.current_time / 120) + 0.5

        requested_time = min(self.time_increment * time_modifier * q_modifier, 40)
        if requested_time <= self.current_time - 4:
            time_to_move = requested_time - 4
        else:
            time_to_move = 2

        best_move, _, _, q = self.engine.best_move(time_limit=int(time_to_move * 1000), node_count=199_000_000, max_sims=self.max_sims, temp=self.temperature)
        self.engine.make_move(best_move)
        self.move_records.append(best_move)
        self.current_q = q
        self.game.make_move(best_move)
        server_move = parsing.ptn_to_server(best_move)
        send(f"Game#{self.game_no} {server_move}")
        print('game_favorability', q)

    def handle_event(self, line: str):
        """
        if receives opponent move, take it and then send it's own move

        it if recives a chat message, it will check if it is a valid command and run it if so
        :param line: the line to handle
        :return: the line to return
        """
        super().handle_event(line)
        if line.startswith(f"Game#{self.game_no}"):
            sent_type = line.split(" ")[1]
            rest = line[len(f"Game#{self.game_no} "):]
            if sent_type in ['P', 'M']:
                ptn_move = parsing.server_to_ptn(rest)
                self.game.make_move(ptn_move)
                self.engine.make_move(ptn_move)
                self.move_records.append(ptn_move)
                self.send_move()
            elif sent_type == "Time":
                self.current_time = int(line.split(" ")[2])
            elif sent_type == "Over":
                self.game_over = True
        elif line.startswith("Tell <"):
            parts = line.split(" ")
            sender_name = parts[1][1:-1]
            message = " ".join(parts[2:]).lower()
            if message.startswith("roadbuilder "):
                message_parts = message.split(" ")
                if len(message_parts) < 2:
                    send(f"Tell {sender_name} malformed input")
                if message_parts[1] == 'set':
                    if len(message_parts) != 4:
                        send(f"Tell {sender_name} malformed input")
                        return self
                    if message_parts[2] == 'c_puct':
                        if not is_float(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (0 <= float(message_parts[3]) <= 5):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.c_puct = float(message_parts[3])
                        self.difficulty = 'Custom'
                        self.restart_engine()
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature, self.difficulty)
                        send(f"Tell {sender_name} c_punt set to {message_parts[3]}")
                    elif message_parts[2] == 'urgency':
                        if not is_float(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (-2 <= float(message_parts[3]) <= 2):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.urgency = float(message_parts[3])
                        self.difficulty = 'Custom'
                        self.restart_engine()
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} urgency set to {message_parts[3]}")
                    elif message_parts[2] == 'alpha':
                        if not is_float(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (0.0001 <= float(message_parts[3]) <= 1):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.alpha = float(message_parts[3])
                        self.difficulty = 'Custom'
                        self.restart_engine()
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} alpha set to {message_parts[3]}")
                    elif message_parts[2] == 'eps':
                        if not is_float(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (0 <= float(message_parts[3]) <= 1):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.eps = float(message_parts[3])
                        self.difficulty = 'Custom'
                        self.restart_engine()
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} eps set to {message_parts[3]}")
                    elif message_parts[2] == 'max_sims':
                        if not is_int(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (0 <= int(message_parts[3])):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.max_sims = int(message_parts[3])
                        self.difficulty = 'Custom'
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} max_sims set to {message_parts[3]}")
                    elif message_parts[2] == 'temperature':
                        if not is_float(message_parts[3]):
                            send(f"Tell {sender_name} {message_parts[3]} could not be converted to float")
                            return self
                        if not (0 <= float(message_parts[3])):
                            send(f'Tell {sender_name} {message_parts[3]} is out of range')
                            return self
                        self.temperature = float(message_parts[3])
                        self.difficulty = 'Custom'
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} temperature set to {message_parts[3]}")
                    elif message_parts[2] == 'difficulty':
                        self.difficulty = message_parts[3]
                        self.c_puct = DIFFICULTY_SETTINGS[message_parts[3]]['c_puct']
                        self.urgency = DIFFICULTY_SETTINGS[message_parts[3]]['urgency']
                        self.alpha = DIFFICULTY_SETTINGS[message_parts[3]]['alpha']
                        self.eps = DIFFICULTY_SETTINGS[message_parts[3]]['eps']
                        self.max_sims = DIFFICULTY_SETTINGS[message_parts[3]]['max_sims']
                        self.temperature = DIFFICULTY_SETTINGS[message_parts[3]]['temperature']
                        self.restart_engine()
                        save_settings(self.c_puct, self.urgency, self.alpha, self.eps, self.max_sims, self.temperature,
                                      self.difficulty)
                        send(f"Tell {sender_name} difficulty set to {message_parts[3]}")
                    else:
                        send(f'Tell {sender_name} unknown setting "{message_parts[2]}"')
                elif message_parts[1] == 'help':
                    send(f'Tell {sender_name} https://github.com/cterry311/TakRoadBuilder/blob/master/RoadBuilderBot_settings.md')
                elif message_parts[1] == 'info':
                    send(f'Tell {sender_name} c_puct {self.c_puct}, urgency {self.urgency}, alpha {self.alpha}, eps {self.eps}, max_sims {self.max_sims}, temperature {self.temperature}, difficulty {self.difficulty}')
                else:
                    send(f'Tell {sender_name} unknown command "{message_parts[1]}"')
        return self

    def restart_engine(self):
        """
        restart the roadbuilder engine with new settings, load all moves to get the engine to match current game state
        :return:
        """
        self.engine = RoadBuilderGame('model_checkpoints/version_14_36.pt',
                                      capacity=200_000_000,
                                      c_puct=self.c_puct,
                                      urgency=self.urgency,
                                      alpha=self.alpha,
                                      eps=self.eps)
        for move in self.move_records:
            self.engine.make_move(move)



    def run_idle(self):
        """
        bake the engine, use opponents time to continue search
        :return:
        """
        if self.game_over:
            return self.future_handler
        else:
            self.engine.bake_tree(time_limit=500, node_count=199_000_000, max_sims=self.max_sims)
            return self


def load_settings():
    """
    load the difficulty settings from settings.json

    :return:
    """
    with open("settings.json", "r") as f:
        data = json.load(f)
        return data

def save_settings(c_puct: float, urgency: float, alpha: float, eps: float, max_sims: int, temperature: float, difficulty: str):
    """
    write difficultly settings to settings.json

    :param c_puct: how much the engine prioritizes the initial policy
    :param urgency: how much the engine prioritizes unexplored nodes
    :param alpha: how spiky the noise is applied to the root, small values is more spiky
    :param eps: how much of the root prior is composed of noise, 0 is no noise, 1 is all noise
    :param max_sims: the maximum number of positions to evaluate, 0 for unbounded
    :param temperature: the temperature with which to sample moves
    :param difficulty: the string describing difficultly
    :return:
    """
    with open("settings.json", "w") as f:
        f.write(json.dumps({
            "c_puct": c_puct,
            "urgency": urgency,
            "alpha": alpha,
            "eps": eps,
            "max_sims": max_sims,
            "temperature": temperature,
            "difficulty": difficulty
        }, indent=4))

def is_int(s: str):
    """
    checks if a string is an int or not
    :param s: the string to check
    :return:
    """
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s: str):
    """
    checks if a string is a float or not
    :param s: the string to check
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


#save_settings(1.5, -0.2, 0.1, 0.25, 0, 0, 'max')
action = SeekEventHandler(600, 10, 2, "A", 0)
#action = AcceptEventHandler("Tiltak_Bot")
handler: EventHandler = LoginEventHandler(os.getenv('PLAYTAK_USERNAME'), os.getenv('PASSWORD'), action.clone())
while True:

    handler = handler.run_idle()

    if handler is None:
        break

    while True:
        try:
            line = lines.get_nowait()
            handler = handler.handle_event(line)
        except queue.Empty:
            break



# medium is beginner
# hard is medium
# expert is hard, except tempature is messing it up sometimes


'''

game_no = None
my_color = None

engine = RoadBuilder('model_checkpoints/version_9_34.pt')
game = tak_engine.Tak(komi=2)
sent_accsept = False


for raw in f_read:                       # iterates line by line; ends at EOF
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