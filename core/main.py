from model.parser import args
from model.train import train_model
from model.generate import generate_from

if(args.generate):
    text = generate_from(args.from_model,args.start_text, args.n_text)
    print(text)
elif(args.train):
    train_model(args)
else:
    print('No valid configuration, use --help.')


