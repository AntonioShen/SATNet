# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rate_d': 2e-4,# Learning rate.
    'learning_rate_g': 1e-3,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 5,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'MNIST',# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
    'checkpoint' : None, # 'checkpoint-mnist-9cat-0-100/model_final_MNIST'
}