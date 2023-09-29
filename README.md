# NER-Flair-ONNX
- This code is used to convert and infer sequence-tagger model in ONNX type.
- Move model to `/checkpoints/best-model.pt`
- Save and move tag dictionary to `/tag_dictionary/tag_dictionary.pkl` via:
```
tag_dictionary.save("./tag_dictionary/tag_dictionary.pkl")
```
when traning

### Environment
- To create virtual environment `venv`:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Export ONNX

- To convert embedding to onnx (if finetune embedding):

```
python3 main.py --scenario embedding
```

- To convert sequence tagger model to onnx:
```
python3 main.py --scenario tagger
```

### Testing

- To test onnx:
```
python3 main.py --scenario test
```

### Inference

- To infer:
```
python3 onnx_infer.py
```