# Set the configuration for deep-learning

opt = dict()
opt['mode'] = 'train'
opt['dataroot'] = '/mnt/data/guest0/datasets/cezanne2photo'
opt['batch_size'] = 2
opt['input_nc'] = 3 # input channel
opt['output_nc'] = 3 # ouptut channel
opt['ngf'] = 64 # generator filters in the last conv layer
opt['ndf'] = 64 # discriminator filters in the first conv layer
opt['use_dropout'] = True # select either using dropout layer or not
opt['dataset_mode'] = 'unaligned' # choose how datasets are loaded. order or random
opt['size'] = 286 # to resize image
opt['crop_size'] = 256 # crop-size of image
opt['pool_size'] = 50 # image buffer size to store prev images
opt['learning_rate'] = 0.001 # learning rate
opt['beta1'] = 0.5 # for Adam optimizer
opt['n_workers'] = 2 # parallel
opt['n_epochs'] = 200 # number of epochs
opt['decay_epoch'] = 100 # epoch to start linearly decaying the learning rate to 0
opt['offset'] = 0 # for lambdaLR scheduler, starting epoch
opt['cuda'] = 'cuda:1' # if use gpu, insert gpu id. or not(using cpu), insert None
opt['eval_num'] = 10
# weights for cycle loss
opt['lambda_identity'] = 0.5
opt['lambda_A'] = 10.
opt['lambda_B'] = 10.
