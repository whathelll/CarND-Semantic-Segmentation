#!/bin/bash

if [ $# -ne 2 ]; then
	echo usage: $0 files_or_folders command_to_run
	exit
fi

FILE_FOLDER=$1
COMMAND=$2

while true
do
    # inotifywait -e close_write $FILE_FOLDER
    inotifywait $FILE_FOLDER
    echo go and change a file
    $COMMAND
done
