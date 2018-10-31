import numpy as np
import pickle, gzip, random, time, math
fl = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(fl, encoding="latin1")
fl.close()

# layers from 1,...
# weights for each layer weights[i] i =2,.. - weights from the layer i-1 to layer i
# y[i] i=1,... = the activation of neurons from layer i
# b[l] bias l =1,...
# z[l] l =2,.. the net input = sum(w[l]*y[l-1]+b[l])

# C(w,b) = 1/2n(sum_x(t-y)^2)
# error = C'(zi) for each neuron i

def softmax(x, sum):
    return math.exp(x)/sum

class NeuralNetwork:
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.weights_input_hidden = np.random.normal(0, 1/math.sqrt(self.nInput), size=(self.nInput, self.nHidden))
        self.velocity_weights_input_hidden = np.random.normal(0, 1/math.sqrt(self.nInput), size=(self.nInput, self.nHidden))
        self.weights_hidden_output = np.random.normal(0, 1/math.sqrt(self.nHidden), size=(self.nHidden, self.nOutput))
        self.velocity_weights_hidden_output = np.random.normal(0, 1/math.sqrt(self.nHidden), size=(self.nHidden, self.nOutput))
        self.last_layer_bias = np.random.standard_normal((1,self.nOutput))
        self.hidden_bias = np.random.standard_normal((1,self.nHidden))

    def sigmoidfct(self, x):
        return 1/(1+math.exp(-x))

    def train(self, input_set, target_set, iterations, learningRate, batches, f, regularization_param, momentum):
        exp = np.vectorize(math.exp)
        soft_max = np.vectorize(softmax)
        sigmoid = np.vectorize(self.sigmoidfct)
        while iterations > 0:
            iterations = iterations - 1
            print("Iteration ", iterations)
            count = 0
            batch = 0
            delta, delta_bias = {}, {}
            delta["hidden_output"] = np.zeros((self.nOutput))
            delta_bias["hidden_output"] = np.zeros(( self.nOutput))

            delta_prev, delta_bias_prev = {}, {}
            #used for momentum
            delta_prev["ho"] = 0
            delta_bias_prev["ho"] = 0

            delta_prev["ih"] = 0
            delta_bias_prev["ih"] = 0

            delta["input_hidden"] = np.zeros((self.nHidden))
            delta_bias["input_hidden"] = np.zeros((self.nHidden))

            for input_layer, target in zip(input_set, target_set):
                digit = target
                target = np.zeros(self.nOutput)
                target[digit] = 1
                if count % 10000 == 0:
                    print(count)
                count += 1
                input_layer = np.matrix(input_layer)
                hidden_layer = np.matmul(input_layer, self.weights_input_hidden)
                hidden_layer = np.add(hidden_layer, self.hidden_bias)

                y_hidden_layer = np.matrix(sigmoid(hidden_layer))

                last_layer = np.matmul(y_hidden_layer, self.weights_hidden_output)
                last_layer = np.add(last_layer, self.last_layer_bias)

                sum = np.sum(exp(last_layer))

                y_last_layer = soft_max(last_layer, sum)

                last_layer_error = np.multiply(np.subtract(target, y_last_layer), (-1)/batches)

                #delta

                #gradient

                hoModification = np.dot(np.transpose(np.matrix(y_hidden_layer)),np.matrix(last_layer_error))

                hidden_layer_error = np.multiply(np.dot(last_layer_error,
                                                           np.transpose(self.weights_hidden_output)),
                                                 np.multiply(y_hidden_layer, np.subtract(1, y_hidden_layer)))

                ihModification = np.dot(np.transpose(np.matrix(input_layer)),np.matrix(hidden_layer_error))


                #velocity for momentum


                # adjusting of the weights and biases on batches
                delta_prev["ho"] = np.multiply(hoModification, learningRate)
                # delta_prev["ho"] = np.add(np.multiply(hoModification, learningRate/batches), np.multiply(momentum, delta_prev["ho"]))
                delta["hidden_output"] = np.add(delta["hidden_output"], delta_bias_prev["ho"])

                delta_bias_prev["ho"] = np.multiply(last_layer_error, learningRate)
                # delta_bias_prev["ho"] = np.add(np.multiply(last_layer_error, learningRate/batches), np.multiply(momentum, delta_bias_prev["ho"]))
                delta_bias["hidden_output"] = np.add(delta_bias["hidden_output"], delta_bias_prev["ho"])

                delta_prev["ih"] = np.multiply(ihModification, learningRate)
                # delta_prev["ih"] = np.add(np.multiply(ihModification, learningRate/batches), np.multiply(momentum, delta_prev["ih"]))
                delta["input_hidden"] = np.add(delta["input_hidden"], delta_prev["ih"])

                delta_bias_prev["ih"] =np.multiply(hidden_layer_error, learningRate)
                # delta_bias_prev["ih"] = np.add( np.multiply(hidden_layer_error, learningRate/batches), np.multiply(momentum, delta_bias_prev["ih"]))
                delta_bias["input_hidden"] = np.add(delta_bias["input_hidden"],delta_bias_prev["ih"])



                if count % batches == 0:
                    batch += 1
                    self.velocity_weights_hidden_output = np.subtract(np.multiply(momentum,
                                                                                   self.velocity_weights_hidden_output),
                                                                       delta["hidden_output"])
                    self.velocity_weights_input_hidden = np.subtract(np.multiply(momentum,
                                                                                 self.velocity_weights_input_hidden),
                                                                     delta["input_hidden"])
                    self.weights_hidden_output = np.subtract(np.multiply(self.weights_hidden_output, (1-learningRate*regularization_param/batches)), delta["hidden_output"])

                    self.weights_input_hidden = np.add(self.weights_input_hidden, self.velocity_weights_input_hidden)
                    self.last_layer_bias = np.subtract(np.multiply(self.last_layer_bias, (1-learningRate*regularization_param/batches)), delta_bias["hidden_output"])
                    self.weights_input_hidden = np.subtract(np.multiply(self.weights_input_hidden, (1-learningRate*regularization_param/batches)), delta["input_hidden"])
                    self.weights_input_hidden = np.add(self.weights_input_hidden, self.velocity_weights_input_hidden)
                    self.hidden_bias = np.subtract(np.multiply(self.hidden_bias, (1-learningRate*regularization_param/batches)), delta_bias["input_hidden"])

    def predict(self, x):
        hidden_layer = np.matmul(x, self.weights_input_hidden)
        hidden_layer = np.add(hidden_layer, self.hidden_bias)
        sigmoid = np.vectorize(self.sigmoidfct)
        soft_max = np.vectorize(softmax)
        exp = np.vectorize(math.exp)

        y_hidden_layer = sigmoid(hidden_layer)
        last_layer = np.matmul(y_hidden_layer, self.weights_hidden_output)
        last_layer = np.add(last_layer, self.last_layer_bias)

        sum = np.sum(exp(last_layer))

        y_last_layer = soft_max(last_layer, sum)

        return np.argmax(y_last_layer)

    def test(self, test_set):
        accuracy, total = 0, 0
        for x, target in test_set:
            total += 1
            digit = self.predict(x)
            if digit == target:
                accuracy += 1
        return accuracy * 100 / total



f = open("nnregularization-"+ time.strftime("%Y-%m-%d-%H-%M-%S"),"w")
x,y = train_set
running_time = time.clock()
nn = NeuralNetwork(784, 100, 10)
iterations = 1
learning_rate = 0.001
batches = 100
xt, yt = test_set
regularization = 3
momentum = 0.4
nn.train(x,y, iterations, learning_rate, batches, f,
         regularization_param=regularization, momentum=momentum )
running_time = time.clock() - running_time
print("Iterations ", iterations, file=f)
print("Learning_rate ", learning_rate, file=f)
print("Reg rate ", regularization, file=f)
print("Momentum param ", momentum, file=f)
print("Batches ", batches, file=f)
print("Training took ", running_time, "s")
print("Training took ", running_time/60, "min")
print("Training took ", running_time, "s", file=f)
print("Training took ", running_time/60, "min", file=f)

accuracy = nn.test(zip(xt,yt))
print(accuracy, "%")
running_time = time.clock() - running_time
print("Testing took ", running_time, "s")
print("Testing took ", running_time/60, "min")
print("Testing took ", running_time, "s", file=f)
print("Testing took ", running_time/60, "min", file=f)
print("Accuracy ", accuracy, "%", file=f)
f.close()