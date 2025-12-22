#!/bin/bash

#uv run python segment.py --help
#uv run python segment.py stage1 --help

for f in $(cat /tmp/inputs.txt)
do
	echo
	echo
	echo
	echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
	echo "$f"
	echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

	uv run python segment.py stage1 -o output-new "$f"
	
done