import time

class Logger:

    def __init__(self, should_store=True,should_print=True):

        self.file_name = "output-{}.txt".format(time.strftime("%H:%M:%S"))
        self.should_store = should_store
        self.should_print = should_print

        with open(self.file_name,'w') as f:
            f.write('')

    def get_model_definition(self, args, r, n_batches):

        start = time.strftime("%H:%M:%S")

        return "================" \
                   "\n{}" \
                   "\nSequence len: {:^8}" \
                   "\nLearning rate: {:^8}" \
                   "\nBatch size: {:^8}" \
                   "\nDropout: {}\n" \
                   "================\n{}\n" \
                   "Data size: {:^8}\n" \
                   "N classes: {:^8}\n" \
                   "N batches: {:^8}\n" \
                   "N epochs: {:^8}\n" \
                   "Start at: {:^8}\n".format(args.layers,args.sequence_len,args.eta,args.batch_size,
                                              args.dropout,self.file_name,len(r.data),r.get_unique_words(),
                                              n_batches,args.epochs,start)

    def get_current_train_info(self, epoch, avg_loss, avg_acc,pred_text,text=''):

        return "\nEpoch: {}\nAVG loss: {}\n " \
               "AVG acc: {}\n " \
               "=====================\n{}\n" \
               "=====================\n{}\n".format(epoch,avg_loss,avg_acc,pred_text,text)

    def log(self,something):
        if(self.should_print):

            print(something)
        if(self.should_store):
            with open(self.file_name, 'a') as f:
                f.write(something)


