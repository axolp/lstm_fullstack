import numpy as np
import json
import copy
from optymalization_methods import *

class LSTM:
    def __init__(self, input_dimension, hidden_dimension,wc= 1, momentum= 0, path= ""):

        if path != "":
            with open(path, "r") as file:
                data = json.load(file)

       
        self.input_dimension= input_dimension
        self.hidden_dimension= hidden_dimension

        self.wx_f = np.array(data['wx_f']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.wh_f = np.array(data['wh_f']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.b_f = np.array(data['b_f']) if path != "" else np.random.randn(hidden_dimension)

        # wagi dla input gate
        self.wx_i = np.array(data['wx_i']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.wh_i = np.array(data['wh_i']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.b_i = np.array(data['b_i']) if path != "" else np.random.randn(hidden_dimension)

        # wagi dla candidate gate
        self.wx_c = np.array(data['wx_c']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.wh_c = np.array(data['wh_c']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.b_c = np.array(data['b_c']) if path != "" else np.random.randn(hidden_dimension)

        # wagi dla output gate
        self.wx_o = np.array(data['wx_o']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.wh_o = np.array(data['wh_o']) if path != "" else np.random.randn(hidden_dimension, input_dimension) * wc
        self.b_o = np.array(data['b_o']) if path != "" else np.random.randn(hidden_dimension)

        self.momentum= momentum
        self.previous_gradients= None

        self.h_prev= float(data["h_prev"]) if path != "" else None
        self.c_prev= float(data["c_prev"]) if path != "" else None

        self.mean= float(data["mean"]) if path != "" else None
        self.std= float(data["std"]) if path != "" else None
    
    def save_to_file(self, filename="model.json"):
        data = {}
        for attr, value in self.__dict__.items():
            print(type(value))
            if isinstance(value, dict):
                continue
            elif isinstance(value, np.ndarray):
                data[attr] = value.tolist()  # Konwersja ndarray na listę
            else:
                data[attr] = value
        
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)


    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Ograniczenie wartości x, aby uniknąć overflow
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        x = np.clip(x, -500, 500)
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    
    def tanh(self, x):
        x = np.clip(x, -500, 500)
        return np.tanh(x)
    
    def dtanh(self, x):
        x = np.clip(x, -500, 500)
        return 1 - np.tanh(x)**2
    
    def forward_propagation(self, x, h_prev, c_prev):

        f_gate= self.sigmoid(
            np.dot( self.wx_f, x ) + 
            np.dot( self.wh_f, h_prev ) + 
            self.b_f
        )
        
        i_gate= self.sigmoid(
            np.dot( self.wx_i, x ) + 
            np.dot( self.wh_i, h_prev ) + 
            self.b_i
        )

        c_gate= self.tanh(
            np.dot( self.wx_c, x ) + 
            np.dot( self.wh_c, h_prev ) + 
            self.b_c
        )

        o_gate= self.sigmoid(
            np.dot( self.wx_o, x ) + 
            np.dot( self.wh_o, h_prev ) + 
            self.b_o
        )
    
        h=  o_gate * self.tanh(c_prev)
        c=  c_prev * f_gate + i_gate * c_gate 

        self.h_prev= h
        self.c_prev= c

        return h, c, (f_gate, i_gate, c_gate, o_gate, c)
    
    def predict(self, previous_price, current_price, c_prev):
        h, _, _ = self.forward_propagation(current_price, previous_price, c_prev)
        prediction= int(h) * self.std + self.mean
        print(h, prediction)
        return prediction
    
    def back_propagation(self, x, h_prev, c_prev, dh_next, dc_next,  states):
        x= x.reshape(-1, 1)

        f_gate, i_gate, c_gate, o_gate, c = states
       


        dc= dc_next + dh_next * o_gate * self.dtanh(c)

        #ide od tylu wiec najpierw output gate
        do= dh_next * self.tanh(c)
        dxo= np.dot( do * self.dsigmoid(o_gate) , x.T )
        dho= np.dot( do * self.dsigmoid(o_gate), h_prev.T )
        dbo= np.sum( do * self.dsigmoid(o_gate), axis=1, keepdims=True )

        #cadidate gate gradients
        dcan= dc * i_gate
        dxcan= np.dot( dcan * self.dtanh(c_gate), x.T )
        dhcan= np.dot( dcan * self.dtanh(c_gate), h_prev.T )
        dbcan= np.sum( dcan * self.dtanh(c_gate), axis=1, keepdims=True)

        #input gate
        di= dc * c_gate
        dxi= np.dot( di * self.dsigmoid(i_gate), x.T )
        dhi= np.dot( di * self.dsigmoid(i_gate), h_prev.T )
        dbi= np.sum( di * self.dsigmoid(i_gate), axis=1, keepdims=True )

        #forget gate 
        df = dc * c_prev
        dxf= np.dot( df * self.dsigmoid(f_gate), x.T )
        dhf= np.dot( df * self.dsigmoid(f_gate), h_prev.T)
        dbf= np.sum( df * self.dsigmoid(f_gate), axis=1, keepdims=True )

        dw = {
            'wx_i': dxi, 'wh_i': dhi, 'b_i': dbi,
            'wx_f': dxf, 'wh_f': dhf, 'b_f': dbf,
            'wx_o': dxo, 'wh_o': dho, 'b_o': dbo,
            'wx_c': dxcan, 'wh_c': dhcan, 'b_c': dbcan
        }
      
        dx= ( 
            np.dot( self.wx_o.T, do ) +
            np.dot( self.wx_c.T, dcan ) +
            np.dot( self.wx_i.T, di ) +
            np.dot( self.wx_f.T, df )   
        ) 

        dh_prev= ( 
            np.dot( self.wh_o.T, do ) +
            np.dot( self.wh_c.T, dcan ) +
            np.dot( self.wh_i.T, di ) +
            np.dot( self.wh_f.T, df )   
        ) 

        dc_prev = f_gate * dc

        return dx, dh_prev, dc_prev, dw

    def update_parameters(self, gradients, lr):


        for param in gradients:
            wages= getattr(self, param)
            gradient= gradients[param]
            
            if self.momentum != 0:
                if self.previous_gradients is not None:
                    previous_gradient= self.previous_gradients[param]
                    setattr( self, param, gradient_descent_with_momentum( wages, gradient, previous_gradient, lr, self.momentum ) )
                else:
                    setattr( self, param, gradient_descent( wages, gradient, lr ) )
            else:
                setattr( self, param, gradient_descent( wages, gradient, lr ) )

        self.previous_gradients= copy.deepcopy(gradients)


    def train(self, data, sequence_length, epochs, lr, n, optymalization_method):

        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std  
        self.mean= mean
        self.std= std

        # Przygotowanie sekwencji i targetów
        sequences, targets = [], []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
            targets.append(data[i+sequence_length])
        sequences = np.array(sequences).reshape(-1, sequence_length, 1)
        targets = np.array(targets).reshape(-1, 1)


        h, c = np.zeros((self.hidden_dimension, 1)), np.zeros((self.hidden_dimension, 1))  # Reset hidden and cell states
        for epoch in range(epochs):
            total_loss = 0
            previous_move= 0
            for i, (seq, target) in enumerate(zip(sequences, targets)):
                if i%n == 0:
                     h, c = np.zeros((self.hidden_dimension, 1)), np.zeros((self.hidden_dimension, 1))

                if optymalization_method == "gradient descent":
                    print("gradient descent")

                elif optymalization_method == "batch gradient descent":
                    # Forward propagation przez całą sekwencję
                    states_list = []
                    for t in range(sequence_length):
                        h, c, states = self.forward_propagation(seq[t].reshape(-1, 1), h, c)
                        states_list.append((h, c, states))
                    
                    # Oblicz stratę dla końcowego kroku
                    loss = (h - target) ** 2
                    total_loss += loss

                    # Backward propagation przez całą sekwencję
                    dh_next = 2 * (h - target)
                    dc_next = np.zeros_like(c)
                    accumulated_gradients = {key: np.zeros_like(value) for key, value in self.back_propagation(seq[0], h, c, dh_next, dc_next, states)[3].items()}
                    for t in reversed(range(sequence_length)):
                        h, c, states = states_list[t]
                        dx, dh_next, dc_next, gradients = self.back_propagation(seq[t].reshape(-1, 1), h, c, dh_next, dc_next, states)
                        for key in gradients:
                            accumulated_gradients[key] += gradients[key]

                    # Aktualizacja parametrów
                    self.update_parameters(accumulated_gradients, lr)
                    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

                elif optymalization_method == "stochastic gradient descent":
                   
                    for t in range(sequence_length):
                        h, c, states = self.forward_propagation(seq[t].reshape(-1, 1), h, c)
                        total_loss+= (h - target) ** 2
                        dh_next = 2 * (h - target)  # Oblicz gradient dla tego kroku
                        dc_next = np.zeros_like(c)

                        dx, dh_next, dc_next, gradients = self.back_propagation(seq[t].reshape(-1, 1), h, c, dh_next, dc_next, states)
                        
                        # Aktualizacja wag po każdym kroku czasowym
                        self.update_parameters(gradients, lr)
     

            print("total loss: ", total_loss)
            print("predykcja: ",  h * std + mean,)
            print(f"dh_next: {dh_next}, dc_next: {dc_next}, dtanh(c): {self.dtanh(c)}")
        
        print("koniec treningu")
        self.save_to_file()
      
                
           
                
            







        

       




    








    