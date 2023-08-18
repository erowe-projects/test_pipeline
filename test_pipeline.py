from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

#Instantiate a model and setup a workflow using a pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)

X_train = ["I've been waiting for a HuggingFace course my whole life.", 
           "I hate HuggingFace, and I don't want to learn."]

res = classifier(X_train)

print(res)

#Display the steps of a tokenizer
sequence = "This course will teach you how to use the HuggingFace API"
res = tokenizer(sequence)
print("Tokenizer Results:")
print(res)
tokens = tokenizer.tokenize(sequence)
print("Tokens:")
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Ids:")
print(ids)
decoded_string = tokenizer.decode(ids)
print("Decoded String:")
print(decoded_string)

#Similar to above, Note: results are in a tensor format. 
# Predictions follow [-1 = negative,1 = postive], 
# and the labels are 1=postive, 0 = negative and they are shown in a list.
batch = tokenizer(X_train, padding = True, truncation = True, max_length = 512, return_tensors="pt")
print("Batch Results:")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print("Outputs:")
    print(outputs)
    print("Predictions:")
    predictions = F.softmax(outputs.logits, dim = 1)
    print(predictions)
    labels = torch.argmax(predictions, dim = 1)
    print("Labels:")
    print(labels)
