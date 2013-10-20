%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function gigmGrad = sigmoidGradient(z)
	gigmGrad = sigmoid(z) .* (1-sigmoid(z));
end