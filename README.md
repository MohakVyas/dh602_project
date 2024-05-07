# Actor critic architecture for XRay image captioning

## Steps:
### Building the vocabulary
```{bash}
python vocab_build.py --caption_path ./clean_indiana_reports.csv --vocab_path ./vocab.pkl
```
### Training (for LSTM based models)
```{bash}
python train_original.py
```
### Training (for transformer models)
```{bash}
python train_modified.py
```
Example usage commands can be found in commands.txt file


