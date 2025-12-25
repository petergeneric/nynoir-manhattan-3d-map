.PHONY: site stl deploy clean

-include Makefile.local

# Generate STL, compress, and deploy to remote server
site: stl deploy

# Generate multiblock.stl.gz from segmentation.json
stl: multiblock.stl.gz

multiblock.stl: segmentation.json
	uv run python generate_stl.py --manifest segmentation.json -o multiblock.stl

multiblock.stl.gz: multiblock.stl
	gzip -kf multiblock.stl

# Deploy compressed STL and webviewer to remote server
deploy: multiblock.stl.gz
	scp multiblock.stl.gz $(REMOTE_HOST):$(REMOTE_PATH)/
	scp webviewer/index.html $(REMOTE_HOST):$(REMOTE_PATH)/

clean:
	rm -f multiblock.stl multiblock.stl.gz
