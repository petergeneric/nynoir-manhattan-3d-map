.PHONY: site stl deploy clean

-include Makefile.local

# Generate STL and deploy to remote server
site: stl deploy

# Generate multiblock.stl from segmentation.json
stl: multiblock.stl

multiblock.stl: segmentation.json
	uv run python generate_stl.py --manifest segmentation.json -o multiblock.stl

# Deploy STL and webviewer to remote server
deploy: multiblock.stl
	scp multiblock.stl $(REMOTE_HOST):$(REMOTE_PATH)/
	scp webviewer/index.html $(REMOTE_HOST):$(REMOTE_PATH)/

clean:
	rm -f multiblock.stl
