run=0

export POL="Linear"
export EXP=5000000

export DIM=3
export N=25
export NN=7
export NT=1000
export DAT=100

for EU in 1000 5000 10000 15000
do
    for R in 32 64 128
    do
        for D in 1 4 16 64
        do 
            for B in 4 16 64
            do 
                export RUN=$run
                export RNN=$R
                export DBS=$D
                export BBS=$B
                export EBRU=$EU
                echo $run
                bash sbatch-irl.sh
                sleep 0.1
                run=$(($run+1))
            done
        done

    done
done
