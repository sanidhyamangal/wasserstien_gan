# Wasseterian GAN
A simple and easy implementation of wasseterian GANs for generation of fashion mnist based images.

![alt Wasseterian GAN](./wasseterian.gif "WGAN")

### requirements
- tensorflow >= 2.0
- pillow
- matplotlib

### How to train model
```
python3 wasseterian.py
```

### How to Generate Images
Image generation can be carried out using two different methods, which are as follows:

### Using shell
```bash
python -i generate_images.py

image_generator = WasseterianImageGenerator()
image_generator.generate_save_images()
```

### In a python file
```python
from generate_images import WasseterianImageGenerator

image_generator = WasseterianImageGenerator()
image_generator.generate_save_images()
```