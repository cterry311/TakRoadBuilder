# model 1
## testing
3.3 loss on test set

1.7 policy loss

1.6 value loss

missing winning move 0.19

made losing move 0.013

## benchmarking
### policy
better than beginner bot, but not massivly so when beginner bot is given 2 sec.

taktition is slightly stronger when given 100 ms

when given 2000 ms
6 wins, 93 losses, 1 draw

tiltak is way stronger, even with only 100 ms.
### value
the value meter is vaugly similar to tiltak, it is still bouncy however and inconsistent.

# model 2
## testing
3.3 avg loss

1.7 policy loss

1.6 value loss

missing winning move 0.19

made losign move 0.013

### policy

taktition 2 sec

wins: 7, losses: 92, draws: 1

### value
the value meter is very bad, it is extremely inconsistent



# model 3
testing model 3_2
-------loss-----------
avg loss 2.0828514327637304
avg policy loss 1.7744781984307896
avg value loss 0.3083732300958694
-------accuracy-----------
avg policy acc 0.448904500554164
avg value acc 0.6670542223194537
-------top move analisis-------------
avg missed winning move 0.2596975673898751
avg made losing move 0.017751860147781982


# model 4
params = {tr_bl=8}


testing model 4_1
-------loss-----------
avg loss 2.0773036790351136
avg policy loss 1.7689629488478835
avg value loss 0.30834073159165276
-------accuracy-----------
avg policy acc 0.4558634113389463
avg value acc 0.6672308541144045
-------top move analisis-------------
avg missed winning move 0.2511505588428665
avg made losing move 0.01701809943101362

depth 3, model 4_2, taktition 2000 ms
wins: 5, losses: 93, draws: 2
total games: 100

### value meter
slightly better than model 1 and 2

# model 5
params = {tr_bl=10}

-------loss-----------
avg loss 2.0798386293478286
avg policy loss 1.76939047411227
avg value loss 0.3104481538549399
-------accuracy-----------
avg policy acc 0.4559785424710843
avg value acc 0.6687841121190653
-------top move analisis-------------
avg missed winning move 0.2445759368836292
avg made losing move 0.015949640843438636


# model 6
params = {tr_ch=128}

-------loss-----------
avg loss 2.0248950394197776
avg policy loss 1.7121470892391266
avg value loss 0.3127479481335265
-------accuracy-----------
avg policy acc 0.47164199441766586
avg value acc 0.66839723112865
-------top move analisis-------------
avg missed winning move 0.19000657462195925
avg made losing move 0.014057310573878119

depth 3, model 6_1, taktition 2000 ms
wins: 6, losses: 92, draws: 2
total games: 100

it seems to have a value prediction that is slightly worse than model 4


week 3 elo estimate, 1500, this was done using the formula 
p = 1 / (1 + exp(c * (e2 - e1)))
where p is the win probability of 1 beating 2, where draws count as 0.5 of a win
c is a constant, I used ln(10) / 400, this is the standard constant used for elo
e2 and e1 are the elo of bots 2 and 1.

I computed my estimate using its games against beginner bot and taktition, using their listed elo on playtak.com
it is only an estimate and probably is not even that accurate. I expect it to have a uncertainty of about +/- 200 points


after implementing search tree, here are the records against beginner bot, taktition, and tiltak

beginner bot
wins: 99, losses: 1, draws: 0
total games: 100


taktition
wins: 83, losses: 16, draws: 1
total games: 100


tiltak
wins: 2, losses: 98, draws: 0
total games: 100

I got pretty incosistent results here, I was abosutly cooked by tiltak, but I won agaisnt taktition with a large ratio
the new elo estimate is now 2000, if I average my results against each of these bots