# PyTorch implementation of beam search decoding for seq2seq models based on https://github.com/shawnwun/NNDIAL.
# Code modified from function "BeamSearchNode" and "beam_decode" obtained from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
import re
import sacrebleu
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from Queue import PriorityQueue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# beam search evaluate

class Beam_Node(object):
    def __init__(self, hidden_s, previousNode, word_id, log_Prob, length):
        
        self.preNode = previousNode
        self.hidden = hidden_s
        self.word_id = word_id
        self.logp = log_Prob
        self.length = length
    
    def log_score(self):
        return self.logp/ float(self.length - 1 + 1e-7) # normalize by the length of the sentence
    
    def __lt__(self, other):
        return self.logp < other.logp # solve the exception

def beam_search_evaluate_sentence(encoder, decoder, testpair_loader):
    
    with torch.no_grad():
        decoded_words = []
        true_words= []
        
        for i, (candidate, length_1, reference, length_2) in enumerate(testpair_loader):
            #batch_size = candidate.size(0)
            #             candidate =  candidate.cuda()
            #             reference = reference.cuda()
            #             encoder_ouputs, encoder_hidden = encoder(candidate, length_1, None)
            
            input_tensor = candidate.cuda()
            reference = reference.cuda()
            encoder_outputs, encoder_hidden = encoder(input_tensor, length_1, None)
            target_length = reference.size()[1]
            
            decoder_hiddens = encoder_hidden
            #           decoder_input = torch.LongTensor([[SOS_token]]*batch_size, device=device).reshape(1,batch_size)
            #             decoder_attentions = torch.zeros(max_length, max_length)
            
            for b_idx in range(batch_size):  #batch size [1, B, D] select sentence from batch
                try:
                    decoder_hidden = decoder_hiddens[:,b_idx, :].unsqueeze(0)
                    encoder_output = encoder_outputs[:,b_idx, :].unsqueeze(1)
                    decoder_input = torch.tensor([[SOS_token]]*1, device = device) #batch_size = 1
                    
                    beam_width = 2
                    topk = 1  #generate 1 sentence as result
                    last_nodes = []
                    # number of sentence +1
                    
                    #initial first node given the decoder input and decoder_hidden
                    node = Beam_Node(decoder_hidden, None, decoder_input, 0, 1)
                    nodes = PriorityQueue() #use PriorityQueue as the main data structure
                    nodes.put((-node.log_score(), node)) # take negative log prob and sort it
                    # start beam search
                    while True:
                        score, nd = nodes.get()
                        decoder_input = nd.word_id
                        decoder_hidden = nd.hidden
                        
                        if nd.preNode != None and nd.word_id.item() == EOS_token:
                            last_nodes.append((score, nd))
                            if len(last_nodes) >= topk:
                                break
                            else:
                                continue
                    
                        #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, 1)
                        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) ##for with out attention method
                        
                        log_prob, indexes = torch.topk(decoder_output, beam_width) # choose the top 5 new hypothesis for next node
                        
                        next_node = []
                        
                        for new_candidate in range(beam_width):
                            decoded_t = indexes[0][new_candidate].view(1, -1)
                            next_log_prob = log_prob[0][new_candidate].item()
                            
                            node = Beam_Node(decoder_hidden, nd, decoded_t, nd.logp + next_log_prob, nd.length+1)
                            # the score is the sum of current node and previous node
                            # the score is the normalized log probability for all nodes in current seuqence
                            # it also set previous node for current node.
                            score = -1*node.log_score() # take the negative log here for the score
                            next_node.append((score, node)) #list of possible next node
                        
                        # put them into queue
                        for i in range(len(next_node)):
                            score, nd = next_node[i]
                            nodes.put((score, nd)) # a priority queue of possible sentence
                except:
                    break

            # choose nbest paths, back trace them
            if len(last_nodes) == 0:
                last_nodes = [nodes.get() for _ in range(topk)] #take two possible sequence from nodes based on the socre
                
                sentence_result = []#candidate
                for score, nd in sorted(last_nodes, key=operator.itemgetter(0)):
                    if nd.word_id.item() == EOS_token:
                        word = [] #possible next word from beam search
                        word.append(train_output_lang.index2word[nd.word_id.item()])
                    
                    while nd.preNode != None:
                        nd = nd.preNode
                        word.append(train_output_lang.index2word[nd.word_id.item()])
                
                    word = word[::-1] #reverse the order of the words
                    words = ' '.join(word)
                    words = re.sub("SOS", '', words) #remove SOS and EOS when output the translation
                    words = re.sub("EOS", '', words)
                sentence_result.append(words)

        decoded_words+= sentence_result
            #             print(len(decoded_words))
            #             print(decoded_words[3])
            true_words += mapback(reference)
#             print(true_words[3])
return decoded_words, true_words


def beam_evaluate(encoder, decoder, loader):
    output_words, true_words = beam_search_evaluate_sentence(encoder, decoder, loader)
    score = sacrebleu.corpus_bleu(output,[true_words])
    return score

