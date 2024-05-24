from transformers import AutoTokenizer

test_string = 'Here is the extracted information in JSON format:\n { \n "Model Number": "YA54", \n  "Brand Name": "SEAPESCA", \n  "Weight": "9g", \n  "Material": "n/a", \n  "Length": "150mm", \n  "Color": "n/a", \n  "Lure Type": "Minnow", \n  "Position": "n/a", \n  "Action": "n/a", \n  "Item": "Fishing Lure", \n  "Quantity": "n/a", \n  "Size": "n/a" \n} \n\nLet me know if you have any further requests!'

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

tokens = tokenizer.tokenize(test_string)
print(len(tokens))
