# Multilingual IPA Phone Recognition Model

## What is this?
This is the base source code to utilize a _Wav2Vec2_-based phone recognition / feature extraction model. It was created in the scope of the PhD thesis [Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech](https://open.fau.de/items/d0c6b800-e217-4049-ab1f-a746fc9b3966) by [Philipp Klumpp](https://scholar.google.com/citations?user=IWvgno4AAAAJ) to analyze pathological speech signals.. The model parameters are deployed via huggingface. Check out the model card [here](https://huggingface.co/pklumpp/Wav2Vec2_CommonPhone).

## Who wants to use this model?
- You want to recognize IPA phones (for example to evaluate pronunciation)
- You want to extract feature vectors and group them by their respective underlying speech sound
- You want to work with multilingual data
- You need a highly robust phone recognizer

This recognizer is capable of predicting phone symbols following the International Phonetic Alphabet (IPA), even for audio recorded under imperfect conditions, such as reverberation, background noise or poor recording equipment (like a smartphone). The model was trained using the multilingual [**Common Phone**](https://zenodo.org/records/5846137) dataset.

For every recognized phone, the model also emits an associated Softmax probability, as well as two feature vectors. The first comes from the CNN block, the second from the last Transformer block.

## Which IPA symbols does this model understand?
Check out `/phonetics/ipa.py` for the full list of IPA symbols.

## Got any numbers?
Sure, the model was evaluated on the test split of **Common Phone**. The following results represent Phone Error Rates (PER) in percent:

| Language | Test PER |
|:---:|:---:|
| English | 11.0 |
| French | 9.9 |
| German | 9.8 |
| Italian | 9.1 |
| Russian | 6.6 |
| Spanish | 8.8 |
| **Average** | **9.2** |

## Quick start
Creating an instance of the model and downloading the parameters is only a single line of code:
```python
    wav2vec2 = Wav2Vec2.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
```

The `example.py` script briefly summarizes the core functions of this model and how to use them. **Always standardize** your inputs before feeding them into the model (see the example)!

## Under what license is this work distributed
Creative Commons Zero 1.0. You can use this model for any purpose, even commercially. See the `LICENSE` for further information.

## How can I reference this work in my publication?

To cite this work, please use the following BibTex snippet:

```
@phdthesis{klumpp2024phdthesis,
  author  = "Philipp Klumpp",
  title   = "Phonetic Transfer Learning from Healthy References for the Analysis of Pathological Speech",
  school  = "Friedrich-Alexander-Universit{\"a}t Erlangen-N{\"u}rnberg",
  address = "Erlangen, Germany",
  year    = 2024,
  month   = may
}
```
