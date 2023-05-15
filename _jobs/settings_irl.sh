# Defaults for Options
EBRU=${EBRU:-10000}     # experiences between reward updates
DBS=${DBS:-16}          # demonstration batch size
BBS=${BBS:-16}          # background batch size
BSS=${BSS:-512}         # background sample size
EXP=${EXP:-5000000}     # number of experiences
RNN=${RNN:-64}          # reward neural net size
POL=${POL:-Linear}      # demo policy type
DIM=${DIM:-3}           # number dimensions
N=${N:-25}              # number fish
NN=${NN:-7}             # number nearest neighbours
NT=${NT:-1000}          # episode length
DAT=${DAT:-100}         # number of data
RUN=${RUN:-0}           # run tag
