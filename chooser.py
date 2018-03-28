import numpy as np

important = 0.0 # important factor
semi = 1.0 # semi important factor
bad = 0.0 # bad factor

def activation_function(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def guess(important, semi, bad):
    inputs = np.array([important, semi, bad])
    weight_input_to_hiden_1 = [0.25, 0.25, 0]
    weights_input_to_hiden_2 = [0.5, -0.4, 0.9]
    weights_input_to_hidden = np.array([weight_input_to_hiden_1,
                                        weights_input_to_hiden_2])

    weights_hiden_to_output = np.array([-1, 1])

    hidden_input = np.dot(weights_input_to_hidden, inputs)
    print("hiden_input: " + str(hidden_input))

    hiden_output = np.array([activation_function(x) for x in hidden_input])
    print("hidden_output: " +str(hiden_output))

    output = np.dot(weights_hiden_to_output, hiden_output)
    print ("output " + str(output))
    return activation_function(output) == 1

print("result: " + str(guess(important, semi, bad)))
