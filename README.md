# Audiffuse
## About
I set out to create a solution to a very subjective problem: Generating Album art Based only off of the audio of a Song. Obviously this is very hard to do, and while the model created is by no means perfect, it is interesting to see what it has learned to generate conditionally off music.

## Model Architecture
The model is a Latent Diffusion model with the conditional head replaced with the [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) Music2Vec model.

<p align="center">
<img src=assets/modelfigure.png />
</p>

In the above diagram this means replacing the T theta with the [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) Music2Vec model.

## Generations
Here is a peice of album art generated by the model, the top left is the original album art from the song and the rest are novel album arts generated by the model.

<p align="center">
<img src=assets/bluesouth.png />
</p>

While obviously music is very subjective, there are some hints here tha tthe model has learned to match the general vibe of a song, from a qualitative perspective all images tend to be very calm ones, and the model has managed to match the color pallete pretty accurately as well, save the bottom right image.

## TODO
- [ ] Release Weights
- [ ] Release How To
