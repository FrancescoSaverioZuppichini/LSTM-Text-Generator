# LSTM Text Generator
## Francesco Saverio Zuppichini

## Quick start

Be use you have Tensorflow installed.

To use a trained model:

```
python3 main.py --generate=True --from_model='shakespeare' --n_text=1000 --start_text="SIR "
```
To train from scratch

```
python3 main.py --train=True --file='data/bible.txt' --layers=512,512
```
Are you in trouble?

```
python3 main,py --help
```
## Results
Here some generated text
```
SCENE	In the court of Wiles.


	[Enter KING EDWARD IV and the BASTARD OF ORLEANS]

HOTSPUR	The king is so too late to be the king.

KING HENRY VI	And that the country fool is strong and sort
	To strike a letter to his princely liege.

KING EDWARD IV	The king is to their hearts to this discourse,
	And to the court of this too long as they
	Will be a maid that shall be through a soldier.

KING EDWARD IV	What, art thou so? to thee and me the king?
	What shall we stand, my lord? why should I say?

KING HENRY IV	The king is so much marched with a cord.

KING HENRY VI	Why, this, my lord, that I did love the king,
	As I am glad of this, that I do love
	The sea of my dear father's son and me.
	What, with this son of England is the king?

CATESBY	There is a mother to the cardinal's head.
```