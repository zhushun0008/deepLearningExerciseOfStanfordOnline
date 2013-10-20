function sigm = sigmoid(z)
    sigm = 1.0 ./(1 + exp(-z));
end