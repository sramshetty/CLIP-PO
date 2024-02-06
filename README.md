# CLIP-RL
Tuning CLIP with RL


## Method
Generative

Text -> Image
1. Sample images using prompt
2. Generate/augment captions for sampled images that contains descriptive differences (spatial locations, orientations, object summary, etc.)
3. Rank these sampled image and generated caption pairs for each prompt

Human-Feedback
- TODO

## Goals:
- [X] Collect ranking dataset
    - [x] Source textual prompts for image retrieval (use the captions themselves?)
    - [x] Source visual prompts for textual retrieval (use the captions themselves?)
        - If locked encoder tuning can apply to either encoder
    - [x] Retrieve 4-9 samples for each prompt
        - Get a few using similarity threshold
        - Add more samples via augmentations
    - [X] Text -> Image: Dense caption samples
- [X] Collect/Create rankings
- [X] Train reward model
- [ ] Implement RL Trainer
    - [ ] Formulate as a DPO problem?
- [ ] Finetune CLIP variants and benchmark

## Citations

```bibtex
@software{Ilharco_OpenCLIP_2021,
    author = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
    doi = {10.5281/zenodo.5143773},
    month = jul,
    title = {{OpenCLIP}},
    version = {v0.1},
    year = {2021}
}
```
