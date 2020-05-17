class tokenizer:
    def __init__(self , name_len):
        self.vocab =  list(' abcdefghijklmnopqrstuvwxyz')
        self.len_v = len(self.vocab)
        self.nl = name_len
        self.init_token()

    def init_token(self) :
        vectors = [[0]*i + [1] + [0]*(self.len_v-i-1)
                        for i in range(self.len_v)]
        self.v_dict = dict(zip(self.vocab , vectors))

    def tkniz(self , name):
        name = name.lower()
        len_n = len(name)
        if len_n > self.nl : 
            name = name[len_n - self.nl:]
        token = [[0] * self.len_v] * (self.nl - len_n)
        for i in name : 
            try : 
                token.append(self.v_dict[i])
            except KeyError :
                token.append([0]*27)
        return token
