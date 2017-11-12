from parser import args
from generate import generate_from
from train import train_model

if(args.generate):
    generate_from(args.from_model,args.start_text, args.n_text)
elif(args.train):
    train_model(args)
else:
    print('No valid configuration, use --help.')


