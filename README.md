# Roadbuilder
this project is an implementation of an alphazero type model plus a Monte Carlo tree search algorithm, trained to play the game of Tak. 
it currently only supports 6x6 size board and is tuned for komi 2, though it can play with any komi.
in a 200 game match against tiltak, it won 190, drew 2 and lost 8. this puts it at approximately +530 elo to tiltak.
in a 200 game match against topaz, it won 13, drew 21, and lost 166. this puts it at approximately -350 elo to topaz.
in both matches, each side was given 2 seconds of movetime.

## requirements
- python 3.12 or newer
- NVIDIA GPU with a compatible driver
- [CUDA Toolkit 13.0](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 9.0](https://developer.nvidia.com/cudnn) or newer for CUDA 13
- packages listed in tei/requirements.txt

running training requires additional packages listed in requirements.txt as well as a GPU installation of pytorch
## building
building the tei interface requires a few steps
1. move into the tei directory from the root `cd tei`
2. create a temporary build enviorment `python3 -m venv build_env`
3. activate the virtual environment `./build_env/scripts/activate`
4. install the requirements `pip install -r requirements.txt`
5. point MODEL_PATH with an absolute path to the model you want to load
6. run build command `pyinstaller --clean RoadBuilder.spec`

running the exe will cause the program to launch in tei mode
## project structure
- benchmarking: contains scripts for testing model performance
- colab: contains Jupyter notebooks for training the model on Google Colab
- engine_clients: contains interfaces for interacting with external engines compiled for testing such as tiltak and topaz
- model: contains the base model class
- model_releases: contains the .onnx released models, compiled to onnx runtime
- playtak_connection: contains code for connecting to playtak.com and seeking games, runs on torch and not onnx runtime
- python_game: contains outdated code for a tak implementation in python
- renforcment_learning: contrary to the name, it does in fact not contain scripts for reinforcement learning, instead it contains scripts for self-play and training against strong opponents
- rust_game: contains an implementation of the game tak, in rust, as well as a Monte Carlo search tree
- self_play: a script for having a model play against itself and save the games to a database
- supervised_training: a script for training a model against example games
- tei: contains the tei interface for the engine
- visualizer: contains a basic viewer for ptn files
## Engine Details
the model was initially trained on playtak.com game archives, trained to predict moves and game outcomes when provided with positions. 
rotational and reflective symmetries were used to increase the number of training examples by roughly 8x.

the model was then trained via self-play, trained to predict game outcomes as well as having its policy output match the search tree enhanced policy

games against strong engines were also played to fill out blind spots.
## Credits
the engine was based on an alphazero implementation and used Monte Carlo search trees, the following papers were used during research
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815)
- [Mastering the game of Go without human knowledge](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/nature-go.pdf)
- [Accelerated Self-Play Learning In Go](https://arxiv.org/pdf/1902.10565)

Prexisting Tak Engines were used as benchmarking opponents, as well as training opponents, the following engines were used for the purposes of training and testing in this project
- [BeginnerBot](https://github.com/NoHatCoder/BeginnerBot)
- [Taktician](https://github.com/nelhage/taktician)
- [Tiltak](https://github.com/MortenLohne/tiltak)
- [TakZero](https://github.com/ViliamVadocz/takzero)
- [Topaz](https://github.com/Jakur/topaz-tak)
- [Takson](https://github.com/Ciekce/takson)
- [Syntaks](https://github.com/Ciekce/syntaks)

initial training data was obtained from [playtak.com](http://playtak.com) game archives

Claude Was used as a resource both for writing code and for Research