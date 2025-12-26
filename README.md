# LaMa inpainting

This appears to do a pretty good job removing text:

```
uv run python inpaint_text.py --no-cache -i ../media/v1/p2.jp2 --text-threshold 0.3 --link-threshold 0.1 -o inpainted-lama.text0.3.link0.1.png
```


The inpainting quality is excellent. We need something better than EasyOCR

IDEA: SAM3 for detecting numbers?



# Text Detection

## EasyOCR
Large bounding boxes. Does an OK job with regular text, not great with very dense text


## DBNET++ model directly

Does an excellent job! Fast!

```
uv run python inpaint_text.py -i ../media/v1/p2.jp2 --expand 0 --threshold 0.09 -o inpainted-dbnet0.3.png
```

### Improvement?

I wonder if it would help if we did straight line detection and articially depressed the heatmap for a few pixels around it? There is a lot of text directly touching the lines, though, so this may not help...

## CRAFT

Tested, even at extreme threshold of 0.01 it still left a lot of text, and expanded to include a lot of map lines

