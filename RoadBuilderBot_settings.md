# RoadBuilderBot Playtak Settings
## basics
- send a command to RoadBuilderBot by sending a message to it prefixed with roadbuilder
- the format for a command is as follows "roadbuilder \<command> \<command option>"
- the most common command is setting the difficulty "roadbuilder set difficulty \<difficulty>"
### difficulty options
- beginner - easiest setting
- easy
- medium
- hard
- expert
- master
- max - the hardest setting, engine running at full strength
## RoadBuilderBot Commands
- ### set
    this command is used for setting the settins of the bot, here are the options
    - difficulty - sets the difficulty of the bot, will set all other settings depending on input
    - c_puct - sets how much inital move predictions influence tree exploration. type float min 0 max 5
    - urgency - influenes how much the bot will explore unexplored regions of the tree. type float min -2 max 2
    - alpha - how spread out the noise is in the tree search, low values lead to spiky noise. type float min 0.0001 max 1
    - eps - how much noise is applied to the trees root. type float min 0 max 1
    - max_sims - the maximum number of simulations the engine is allowed to run, 0 is unlimited. type int min 0
    - temperature - how much randomness to apply to the move selection post search, 0 is selection of the best move. type float min 0
- ### help
    this command will provide a link to the github page describing the commands
- ### info
    this command will display the current config of the bot