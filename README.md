# Actor critic architecture for XRay image captioning

## Steps:
### Building the vocabulary
```{bash}
python vocab_build.py --caption_path ./clean_indiana_reports.csv --vocab_path ./vocab.pkl
```
### Training
```{bash}
python train.py
```
