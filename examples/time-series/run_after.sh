#!/bin/bash

# Run a command again after a certain process (with known PID given as argument) has finished. The command to run is "nohup python main_wsm.py > nohup.log &"
 
#  The script  run_after.sh  takes a PID as an argument and checks if the process with that PID exists. If it does, the script waits until the process has finished. After the process has finished, the script runs the command  nohup python main_wsm.py > nohup.log & 
#  The script can be run as follows: 
#  $ ./run_after.sh 12345
 
#  where  12345  is the PID of the process that the script should wait for. 
#  Conclusion 
 


# Check if the PID is given as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

# Check if the process with the given PID exists
if ! ps -p $1 > /dev/null; then
    echo "Process with PID $1 does not exist."
    exit 1
fi

# Wait until the process with the given PID has finished
while ps -p $1 > /dev/null; do
    sleep 1
done

# Run the command
nohup python main_wsm.py > nohup.log &
