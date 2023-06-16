# Scene-Creation-using-Image-Outpainting

We aim to build a stable diffusion model and evaluate it. Furthermore,
to demonstrate a practical use of the model we employ it for the task of image
outpainting. We can use the diffusion model to generate an entire scene by making
use of reference pixels generated in the original image. This can help us control
minute details in the scene as we would be able to specify what particular area of 
the image is to be modified using text prompts.

There are 2 main files in the repository -
1. `Textual Inversion (Training).py` - 
To create customized scenes specific to an entity, it is required to train the model in such a way that
for a given token, a set of desired visuals are learned through embeddings. Textual Inversion is a method used to extract new ideas from a limited set of example images. In this file, we have added a new token 'Geisel Library' and trained the model to get custom embeddings for the same. The embeddings learned can be found in the `learned-geisel` folder. 

2. `Image Outpainting (Inference).ipynb` -
Once we are able to generate customised images for Geisel, our next step is generate its surroundings
guided by user given text prompts. We have tried to outpaint and extend the surroundings of Geisel Library using the embeddings trained previously.

Also, we have collected a few images of 'Geisel Library' and trained the model on this custom dataset. This can be found in the `data` folder.

