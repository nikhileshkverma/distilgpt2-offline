Sure! Here’s the full **README.md** in one code block for easy copy-paste on GitHub:

````markdown
# DistilGPT-2 Offline Setup and Usage

This repository shows how to set up and run the DistilGPT-2 model offline with PyTorch and Hugging Face Transformers.

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Setup Environment](#setup-environment)  
- [Install Required Packages](#install-required-packages)  
- [Download and Save Model Locally](#download-and-save-model-locally)  
- [Run Offline Model](#run-offline-model)  
- [Common Errors & Fixes](#common-errors--fixes)  
- [Push to GitHub](#push-to-github)  

---

## Prerequisites

- Ubuntu or Linux-based OS (tested on Ubuntu 22.04+)  
- Python 3.10  
- Git installed  

---

## Setup Environment

```bash
mkdir distilgpt2_offline && cd distilgpt2_offline
python3 -m venv venv
source venv/bin/activate
````

---

## Install Required Packages

```bash
pip install --upgrade pip
pip install torch==2.2.2+cu121 numpy==1.26.4 transformers==4.22.0
```

> **Note:**
>
> * Downgrade numpy to 1.26.4 to fix `_ARRAY_API not found` warning
> * Install matching Torch version for CUDA 12.1 support

---

## Download and Save Model Locally

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained("./distilgpt2")
model.save_pretrained("./distilgpt2")
```

---

## Run Offline Model

Create `offline_run.py` with:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./distilgpt2")
model = AutoModelForCausalLM.from_pretrained("./distilgpt2")

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=inputs['attention_mask']
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Run:

```bash
python offline_run.py
```

---

## Common Errors & Fixes

* **\_ARRAY\_API not found warning**
  Fix by downgrading numpy:

  ```bash
  pip install numpy==1.26.4
  ```

* **Attention mask and pad token id warning**
  Always pass `attention_mask` and `pad_token_id` in generation (see code above).

* **GitHub push 403 error**
  Use a GitHub Personal Access Token (PAT) for HTTPS authentication or set up SSH keys.
  See: [https://docs.github.com/en/authentication](https://docs.github.com/en/authentication)

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial offline DistilGPT-2 setup"
git branch -M main
git remote add origin https://github.com/nikhileshkverma/distilgpt2-offline.git
git push -u origin main
```

> Enter your GitHub Personal Access Token when prompted for password.

---

## License

MIT License

---

*Created by Nikhilesh K Verma*

```

Just copy-paste this entire block into your `README.md` file on GitHub or locally!
```
