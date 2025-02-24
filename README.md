# 🌍 Translation using mBART

This project demonstrates how to use the `mBART` model from Hugging Face's `transformers` library to translate text between different languages.

## 🚀 Features

- Utilizes `facebook/mbart-large-50-many-to-many-mmt` model for multilingual translation.
- Supports over 50 languages.
- Implements a simple Python function to translate text using `MBartForConditionalGeneration`.

## 📥 Installation

To use this project, install the required dependencies:

```bash
pip install transformers torch
```

## 🛠 Usage

### Load the Model and Tokenizer

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
```

### Define Translation Function

```python
def translate_text(text, src_lang="en_XX", tgt_lang="fa_IR"):
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text
```

### Example Usage

```python
text = "Once there was a race going on. A tortoise and a Hare participated..."
translated_text = translate_text(text)
print(translated_text)
```

## 🌟 Supported Languages

The `mBART` model supports over 50 languages, including:

- English (`en_XX`)
- Persian (`fa_IR`)
- French (`fr_XX`)
- German (`de_DE`)
- Spanish (`es_XX`)

For the full list of supported languages, check the Hugging Face Model Page.

---

Feel free to contribute or modify the code to fit your needs! 😊

