#!/bin/bash

#uv run python segment.py --help
#uv run python segment.py stage1 --help

for f in ../media/v2/*.jp2
do
	echo
	echo
	echo
	echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
	echo "$f"
	echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

	uv run python segment.py stage1 --no-overwrite -o output "$f"	
done