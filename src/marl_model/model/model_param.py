class SimpleModelParam:

    learning_rate = 0.001               # Learning rate for DNN training
    batch_size = 32                     # Batch size for DNN training
    epochs = 1                          # Number of epochs for DNN training
    gamma = 0.95                        # Reinforcement learning discount ratio
    min_epsilon = 0.05                  # Minimum value for epsilon
    epsilon_end_frame = 50000           # How many frames need to decay epsilon to minimum
    replay_memory_size = 1000000        # History buffer length
    train_frame_rate = 4                # Model training rate
    target_update_frame_rate = 1000     # Target model update rate

class AtariModelParam:

    learning_rate = 0.001
    gamma = 0.99
    batch_size = 32
    min_epsilon = 0.1
    epsilon_end_frame = 1000000
    epochs = 1
    replay_memory_size = 200000
    train_frame_rate = 4
    target_update_frame_rate = 10000

class ClassicControlModelParam:

    learning_rate = 0.001
    gamma = 0.99
    batch_size = 32
    min_epsilon = 0.05
    epsilon_end_frame = 10000
    epochs = 1
    replay_memory_size = 10000
    train_frame_rate = 4
    target_update_frame_rate = 1000
