# This is the main module of the project, it contains the StudentModel class which uses all the others to form a working model.
# I designed StudentModel as an 'abstraction'. Its fields contain the dataset, the textprocessor (and vocabulary),
# the device, the real model ... He takes care of instanzation, training, predict and so on.

import numpy as np
from typing import *
from model import Model

import json
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from stud.Vocabulary import Vocabulary
from stud.TextDataset import TextDataset
from stud.NeuralNetworks import MLPClassifier, LSTMClassifier
from stud.TextProcessor import TextProcessor



# Build and load the 'best' model in ram from disk (to predict the labels of the test set)
def build_model(device: str) -> Model:

    # Paths defined from the training stage (notebook)
    arch_type= "LSTM"       # can be replaced with 'MLP'
    weights_path = "./model/LSTM_50D_23300TOK.pth"
    embedding_path = "./model/embedding_50D_23300TOK.json"
    train_path = "./data/train.jsonl"
    val_path = "./data/dev.jsonl"
    vocab_path = "./model/Vocabulary_23300TOK.json"

    return StudentModel.from_disk(arch_type, weights_path, embedding_path, train_path, val_path, vocab_path, device)



class StudentModel(Model):

    def __init__(self, classifier: torch.nn.Module, dataset: TextDataset, text_processor:TextProcessor,
                 device:torch.DeviceObjType):
        self.dataset = dataset
        self.device = device
        self.text_processor = text_processor
        self.classifier = classifier
        self.classifier.to(self.device)


    # Train the classifier, returns the logs, and reload the best weights founded.
    def train_network(self, learning_rate: float=0.001, batch_size: int=32,
                            num_epoch: int=50, verbose: bool=True) -> Tuple[Dict, Dict]:

        train_log = {'prec':[],'recall':[],'f1':[],'acc':[],'loss':[]};
        val_log   = {'prec':[],'recall':[],'f1':[],'acc':[],'loss':[]};

        np.random.seed(1337); torch.manual_seed(1337)
        if self.device == "cuda":   torch.cuda.manual_seed_all(1337)

        optimizer = self.classifier.get_optimizer(learning_rate)
        scheduler = self.classifier.get_scheduler(optimizer,verbose=verbose)
        loss_fun  = self.classifier.get_loss_fun()
        best_weight = None

        for epoch_index in range(num_epoch):

            # --- LOOP TRAINING --- #
            self.dataset.set_split('train')
            self.classifier.train()

            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            batch_scores = []

            for batch_dict in dataloader:
                optimizer.zero_grad()

                x_in, x_len = self.text_processor.batch_to_input(batch_dict['x1_data'], batch_dict['x2_data'],self.device)
                labels = batch_dict['y_target'].to(self.device)

                y_pred = self.classifier(x_in=x_in, x_len=x_len)
                loss = loss_fun(y_pred, labels)

                loss.backward()
                optimizer.step()

                batch_scores += [self.compute_scores(labels.cpu(), torch.sigmoid(y_pred).detach().cpu().numpy(), loss)]

            self.update_log(batch_scores, train_log)


            # --- LOOP VALIDATION --- #
            self.dataset.set_split('val')
            self.classifier.eval()

            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            batch_scores = []

            for batch_dict in dataloader:
                x_in, x_len = self.text_processor.batch_to_input(batch_dict['x1_data'], batch_dict['x2_data'],self.device)
                labels = batch_dict['y_target'].to(self.device)

                y_pred = self.classifier(x_in=x_in, x_len=x_len)
                loss = loss_fun(y_pred, labels)

                batch_scores += [self.compute_scores(labels.cpu(), torch.sigmoid(y_pred).detach().cpu().numpy(), loss)]

            self.update_log(batch_scores, val_log)

            # Save best weigths
            if val_log["loss"][-1] <= min(val_log["loss"]):
                best_weight = copy.deepcopy(self.classifier.state_dict())

            if verbose:
                print("TRAIN: " + str(train_log['loss'][-1]) + " - " + str(train_log['acc'][-1]))
                print("VAL: " + str(val_log['loss'][-1]) + " - " + str(val_log['acc'][-1]))

            # Dinamic learning rate policy + earling stopping (after 3 l.r. changes)
            if scheduler is not None:
                scheduler.step(val_log['loss'][-1])
                if (learning_rate * pow(scheduler.factor, 3) >= optimizer.param_groups[0]['lr']): break;

        # reaload best weights before exiting
        self.classifier.load_state_dict(best_weight)

        return (train_log, val_log)


    # Predict Method (signature corrected)
    def predict(self, sentence_pairs: List[Dict]) -> List[str]:
        self.classifier.eval()
        predictions = []
        for sample in sentence_pairs:
            x_in,x_len = self.text_processor.batch_to_input([sample['sentence1']],[sample['sentence2']],self.device)
            y_pred = self.classifier(x_in=x_in,x_len=x_len, apply_sigmoid=True)
            predictions += ["True" if y_pred.item() >= 0.5 else "False"]
        return predictions


    # Compute the score at each batch
    def compute_scores(self, labels:torch.Tensor, predictions:torch.Tensor, loss:torch.Tensor) -> Tuple:
        predictions = np.array((predictions - 0.5) > 0)
        p = precision_score(labels, predictions, average='macro',zero_division=0)
        r = recall_score(labels, predictions, average='macro',zero_division=0)
        f = f1_score(labels, predictions, average='macro',zero_division=0)
        a = accuracy_score(labels, predictions)
        l = loss.item()
        return (p,r,f,a,l)


    # Update the log at the end of each epoch using batch scores
    def update_log(self, batch_scores:List[Tuple], log:Dict) -> None:
        batch_size = len(batch_scores)
        average_score = [0,0,0,0,0]

        for score in batch_scores:
            for i in range (5):
                average_score[i] += score[i]

        log['prec']    += [average_score[0] / batch_size]
        log['recall']  += [average_score[1] / batch_size]
        log['f1']      += [average_score[2] / batch_size]
        log['acc']     += [average_score[3] / batch_size]
        log['loss']    += [average_score[4] / batch_size]


    # Load the object from disk
    @classmethod
    def from_disk(cls, arch_type:str, weights_path:str, embedding_path:str,
                  train_path:str, val_path:str, vocab_path:str, device_str:str='cpu'):

        try:        dataset = TextDataset.from_disk(train_path, val_path)
        except:     dataset = None

        voc = Vocabulary.from_disk(vocab_path)
        tp  = TextProcessor(voc)
        device = torch.device(device_str)

        with open(embedding_path) as fp:
            embedding = torch.tensor(json.load(fp)).to(device)

        if arch_type == "MLP":
            classifier = MLPClassifier(len(embedding[0]),len(embedding[0]),len(voc),voc.pad_index,embedding)
        if arch_type == "LSTM":
            classifier = LSTMClassifier(len(embedding[0]),len(embedding[0]),len(voc),voc.pad_index,embedding)

        classifier.to(device)
        classifier.eval()
        try:        classifier.load_state_dict(torch.load(weights_path))
        except:     classifier.load_state_dict(torch.load(weights_path, map_location="cpu"))

        return cls(classifier,dataset,tp,device)


    def __str__(self):
        return str(self.classifier)





# ---------------------------------------- #


class RandomBaseline(Model):

    options = [
        ('True', 40000),
        ('False', 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]

