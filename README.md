# CLIP-RL
Tuning CLIP with RL


### Method
Generative

Text -> Image
1. Sample images using prompt
2. Generate/augment captions for sampled images that contains descriptive differences (spatial locations, orientations, object summary, etc.)
3. Rank these sampled image and generated caption pairs for each prompt


Human-Feedback
- TODO

### Goals:
- [ ] Collect ranking dataset
    - [x] Source textual prompts for image retrieval (use the captions themselves?)
    - [x] Source visual prompts for textual retrieval (use the captions themselves?)
        - If locked encoder tuning can apply to either encoder
    - [ ] Retrieve 4-9 samples for each prompt
    = [ ] Text -> Image: Dense caption samples
- [ ] Collect/Create rankings
- [ ] Train reward model
- [ ] Finetune CLIP variants and benchmark