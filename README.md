# BRIA RMBG 1.4 Background Removal

This is a Replicate implementation of BRIA RMBG 1.4 for background removal.

## Model Description

BRIA RMBG 1.4 is an image matting model that can remove backgrounds from images with high quality results.

## Input

- Image file (supports common formats like JPG, PNG)

## Output

- PNG image with transparent background

## Example Usage

```python
import replicate

output = replicate.run(
    "username/rmbg:version",
    input={
        "image": open("input.jpg", "rb")
    }
)
