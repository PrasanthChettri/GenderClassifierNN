from typing import List
class Tokenizer:
    '''
        tokenizer does not take special characters
    '''
    VOCAB = list(' abcdefghijklmnopqrstuvwxyz')
    LEN_OF_VOCAB = len(VOCAB)
    def __init__(self , name_len : int):
        '''
            params :
                name_len : lenght we support for names to be tokenized
        '''

        self.max_len = name_len
        self.init_token()

    def init_token(self)->None :
        '''
            init token intialises to 
            the dictionary for the vocab
        '''
        vectors = []
        for i in range(Tokenizer.LEN_OF_VOCAB):
            vectors.append(
                [0]*i + [1] + [0]*(Tokenizer.LEN_OF_VOCAB-i-1)
            )
        self.v_dict = dict(zip(Tokenizer.VOCAB , vectors))

    def tkniz(self , name : str)->List:
        name = name.lower()
        '''
            filter out symbols not in vocab
        '''
        name = list(
                    filter(lambda word : word in Tokenizer.VOCAB , name)
                )
        len_name = len(name)
        '''
            padding the tokens with zero at the front
            if len_name bigger than len supported 
            we snipp it
        '''
        if len_name < self.max_len : 
            token = [[0] * Tokenizer.LEN_OF_VOCAB] * (self.max_len - len_name)
        else : 
            name = name[:self.max_len]
            token = []

        for alpha in name : 
            '''
                tokenize name 
            '''
            token.append(self.v_dict[alpha])
        return token