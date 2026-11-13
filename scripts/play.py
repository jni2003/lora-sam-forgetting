# scratch script to get familiar with the models on a smaller scale before the experiments


# generate text: sentence completion
# look at all guesses for next word and confidences
# score how well it would predict a given sentence's words given the word before
import torch
from transformers import AutoTokenizer 
from src.convert import load_olmo_as_llama 

toks = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer") # load tokenizer
sam = load_olmo_as_llama("checkpoints/olmo-60m-sam") # load sam 60m model
adamw = load_olmo_as_llama("checkpoints/olmo-60m-adamw") # load adamw 60m model

for name, model in [("sam", sam), ("adamw", adamw)]:
    print(name + ":")
    inputs = toks("The capital of California is", return_tensors="pt") # text -> token ids
    with torch.no_grad(): # don't track gradients unnecessarily
        outputIds = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        print(toks.decode(outputIds[0]))
        
        allScores = model(**inputs).logits
        lastScores = allScores[0, -1] # scores for last position aka guess for next word
        probs = torch.softmax(lastScores, dim=-1) # scores -> 1-summing probabilities
        topValues, topIds =  torch.topk(probs, 5)
        for i in range(len(topValues)):
            print(toks.decode(topIds[i]), ":", topValues[i])
    
        ids = toks("The capital of California is Sacramento", return_tensors="pt")["input_ids"]
        print("loss:", model(input_ids=ids, labels=ids).loss.item())
    
        scrambledIds = toks("Sacramento California is of capital the", return_tensors="pt")["input_ids"]
        print("loss (scrambled):", model(input_ids=scrambledIds, labels=scrambledIds).loss.item())
    
        # code input for fine tuning cmp later
        inputs2 = toks("def add(a, b):\n    return", return_tensors="pt")
        outputIds2 = model.generate(**inputs2, max_new_tokens=20, do_sample=False)
        print(toks.decode(outputIds2[0]))
