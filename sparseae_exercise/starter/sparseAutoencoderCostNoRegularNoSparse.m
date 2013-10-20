function [cost,grad] = sparseAutoencoderCostNoRegularNoSparse(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 


%  (a) Implement forward propagation in your neural network, and implement the 
%      squared error term of the cost function.  Implement backpropagation to 
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
%      to verify that the calculations corresponding to the squared error cost 
%      term are correct.


% 	This is  a first version without regularization or Sparsity.
%	So set lambda and beta equalling to zero respectively.  
Jbasic = 0;
beta = 0;
lambda = 0;
exampleNum = size(data,2);
for num = 1:exampleNum
%	(a).01 Implement forward propagation.
	a1 = data(:,num);
	z2 = W1 * a1 + b1;
	a2 = sigmoid(z2);
	z3 = W2 * a2 + b2;
	a3 = sigmoid(z3);
	Jbasic = Jbasic +(a3-a1)'*(a3-a1);
	
%	(a).02 Implement back propagation.
	delta3 = (a3 - a1) .* a3 .*(1-a3);
	delta2 = W2'*delta3 .*a2.*(1-a2);
	W2grad = W2grad + delta3 * a2';
	b2grad = b2grad + delta3;
	W1grad = W1grad + delta2 * a1';
	b1grad = b1grad + delta2;

end


W2grad = W2grad ./exampleNum + lambda * W2;
W1grad = 1/exampleNum * W1grad + lambda * W1;
b2grad = b2grad ./ exampleNum;
b1grad = b1grad ./ exampleNum;
Jbasic = Jbasic/(2*exampleNum);
regularTerm = lambda/2 *(sum(sum((W1.^2))) +sum(sum((W2.^2))));
cost = Jbasic ;







%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


