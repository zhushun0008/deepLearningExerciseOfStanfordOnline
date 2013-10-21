%%======================================================================
%%======================================================================
%============================ Vectorization ============================
%%======================================================================
%%======================================================================


%%======================================================================
%% STEP 1: load images 
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte

images = loadMNISTImages('train-images.idx3-ubyte');
 
% We are using display_network from the autoencoder code
display_network(images(:,1:100)); % Show the first 100 images

fprintf('programs pause....\n');
fprintf('Showed iamges...\n');
pause;
%%======================================================================
%% STEP 2: Initialize the parameter 

visibleSize = 28*28;   % number of input units 
hiddenSize = 196;     % number of hidden units 
sparsityParam = 0.1;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 3e-3;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term    
patches = images(:,1:10000);	% first 10000 images from the MNIST dataset.


%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath ../sparseae_exercise/starter/minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

							  
							  

%%======================================================================
%% STEP 3: Visualization 
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 

toc;
