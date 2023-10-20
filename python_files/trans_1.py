from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I like that soooo much!")

print(res)
