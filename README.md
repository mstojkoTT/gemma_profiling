# gemma_profiling

## Reproduce steps

```
git clone --recurse-submodules https://github.com/mstojkoTT/transformers_gemma_profile
cd transformers_gemma_profile

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Now put your HF token in `hf_token.txt`

```

./profile.sh
```


Put in the path to the picture in here in `main.py`:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/ttuser/mstojko/gemma_profiling/picture.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
```