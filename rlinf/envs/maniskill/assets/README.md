---
license: odc-by
language:
- en
---
## Asset Download

The assets need to be placed into RLinf's ManiSkill environment folder with the name `assets`.

```bash
# uv pip install huggingface_hub if you don't have it
cd <path_to_RLinf>/rlinf/envs/maniskill
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
```

You can also use `git` to clone the repository:

```bash
cd <path_to_RLinf>/rlinf/envs/maniskill
git clone https://hf.co/datasets/RLinf/maniskill_assets ./assets
```

## License
Our assets are attributed to [objaverse](https://huggingface.co/datasets/allenai/objaverse-xl).
We follow the license of Objaverse-XL. The use of the dataset as a whole is licensed under the ODC-By v1.0 license. Individual objects in Objaverse-XL are licensed under different licenses.