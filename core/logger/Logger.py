import time
import sys

class Logger:
    """
    This class logs shit on stdout and write in a log file
    """
    def __init__(self, should_store=True, should_print=True, file_name="output-{}.txt".format(time.strftime("%H:%M:%S"))):

        self.file_name = file_name
        self.should_store = should_store
        self.should_print = should_print

        with open(self.file_name,'w') as f:
            f.write('')

    def get_model_definition(self, args, r, n_batches):

        start = time.strftime("%H:%M:%S")

        return "================" \
                   "\nModel layout:{}" \
                   "\nData: {}" \
                   "\nSequence len: {:^8}" \
                   "\nLearning rate: {:^8}" \
                   "\nBatch size: {:^8}" \
                   "\nDropout: {}\n" \
                   "================\n{}\n" \
                   "Data size: {:^8}\n" \
                   "N classes: {:^8}\n" \
                   "N batches: {:^8}\n" \
                   "N epochs: {:^8}\n" \
                   "Start at: {:^8}\n".format(args.layers,args.file,args.sequence_len,args.eta,args.batch_size,
                                              args.dropout,self.file_name,len(r.data),r.get_unique_words(),
                                              n_batches,args.epochs,start)

    def get_current_train_info(self, epoch, avg_loss, avg_acc, val_loss, pred_text="", text=""):

        return "\nEpoch: {}\nAVG loss: {}\n" \
               "VAL loss: {}\n" \
               "AVG acc: {}\n" \
               "=====================\n{}\n" \
               "=====================\n{}\n".format(epoch,avg_loss,val_loss,avg_acc,pred_text,text)

    def log(self, something):

        if(self.should_print):
            print(something)

        if(self.should_store):
            with open(self.file_name, 'a', encoding='utf-8') as f:
                f.write(something)


class Progress:
    """
    TODO: not working lol
    """
    @staticmethod
    def progress(size, i, total):
        step = total / size

        if (i % step == 0):
            sys.stdout.write("-")
            sys.stdout.flush()

        if (i == total):
            sys.stdout.write("\n")

