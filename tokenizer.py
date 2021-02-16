from typing import List
class tokenizer:
    '''
        tokenizer does not take special characters
    '''
    VOCAB = list(' abcdefghijklmnopqrstuvwxyz')
    tokenizer.LEN_OF_VOCAB = len(VOCAB)
    def __init__(self , name_len : int):
        '''
            params :
                name_len : lenght we support for names to be tokenized
        '''

        tokenizer.LEN_OF_VOCAB = len(self.vocab)
        self.max_len = name_len
        self.init_token()

    def init_token(self)->None :
        '''
            init token intialises to 
            the dictionary for the vocab
        '''
        vectors = []
        for i in range(tokenizer.LEN_OF_VOCAB):
            vectors.append(
                [[0]*i + [1] + [0]*(tokenizer.LEN_OF_VOCAB-i-1)]
            )
        self.v_dict = dict(zip(tokenizer.VOCAB , vectors))

    def tkniz(self , name : str)->List:
        name = name.lower()
        for alpha in name : 
            if alpha not in tokenizer.VOCAB :  
                raise Exception(f"Word out of vocab for {alpha}")

        len_name = len(name)
        '''
            padding the tokens with zero at the front
            if len_name bigger than len supported 
            we snipp it
        '''
        if len_name > self.max_len : 
            token = [[0] * tokenizer.LEN_OF_VOCAB] * (self.max_len)
        else : 
            token = [[0] * tokenizer.LEN_OF_VOCAB] * (self.max_len - name_len)

        for alpha in name : 
            '''
                tokenize name 
            '''
            token.append(self.v_dict[alpha])
        return token