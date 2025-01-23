from medvlm.models.tokenizer import Tokenizer


tokenizer = Tokenizer()

test = tokenizer("Das ist ein Test")
print(test)
print(tokenizer.decode([test]))

