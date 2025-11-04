import numpy as np
from agent import DeepQLearningAgent, BreadthFirstSearchAgent
from game_environment import Snake
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import os
import sys

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\Marco\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# some global variables
version = 'v17.1'
iteration = 163500

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
max_time_limit = -1

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit)
s = env.reset()
# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                           buffer_size=1, version=version)

# load weights into the agent
agent.load_model(file_path='models/{:s}/'.format(version), iteration=iteration)

'''
# make some moves
for i in range(3):
    env.print_game()
    action = agent.move(s)
    next_s, _, _, _, _ = env.step(action)
    s = next_s.copy()
env.print_game()
'''

# PyTorch hook to capture intermediate layer outputs
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hook for first conv layer
layer_num = 0  # First conv layer
conv_layers = [module for module in agent._model.conv_layers if isinstance(module, nn.Conv2d)]
if conv_layers:
    conv_layers[layer_num].register_forward_hook(get_activation(f'conv{layer_num}'))

# Get intermediate output
s_tensor = torch.FloatTensor(s.reshape(1, board_size, board_size, frames)).to(device)
with torch.no_grad():
    _ = agent._model(s_tensor)
    output_temp = activation[f'conv{layer_num}'].cpu().numpy()[0]  # Shape: (channels, height, width)

print('selected layer shape : ', output_temp.shape)

# save layer weights (PyTorch weights: out_channels, in_channels, height, width)
plt.clf()
w = conv_layers[layer_num].weight.detach().cpu().numpy()  # Shape: (out_channels, in_channels, kernel_h, kernel_w)
nrows, ncols = (w.shape[0]*w.shape[1])//8, 8
fig, axs = plt.subplots(nrows, ncols, figsize=(17, 17))
for i in range(nrows):
    for j in range(ncols):
        filter_idx = i*(ncols//2)+(j//2)
        channel_idx = j%w.shape[1]
        if filter_idx < w.shape[0] and channel_idx < w.shape[1]:
            axs[i][j].imshow(w[filter_idx, channel_idx, :, :], cmap='gray')
        axs[i][j].axis('off')
fig.savefig('images/weight_visual_{:s}_{:04d}_conv{:d}.png'\
            .format(version, iteration, layer_num), 
            dpi=72, bbox_inches='tight')
# sys.exit()

done = 0
t = 0
fig = plt.figure(figsize=(17,17))
while(not done):
    # Get intermediate output for current state
    s_tensor = torch.FloatTensor(s.reshape(1, board_size, board_size, frames)).to(device)
    with torch.no_grad():
        _ = agent._model(s_tensor)
        output_temp = activation[f'conv{layer_num}'].cpu().numpy()[0]  # Shape: (channels, height, width)
    
    # play game
    action = agent.move(s, env.get_legal_moves(), env.get_values())
    next_s, _, done, _, _ = env.step(action)
    
    # visualize weights, we will add the game state as well
    plt.clf()
    nrows, ncols = output_temp.shape[0]//4, 4  # PyTorch: channels first
    
    # add the game image
    ax = plt.subplot(nrows, ncols+2, (1, ncols+2+2))
    ax.imshow(s[:,:,0], cmap='gray')
    ax.set_title('Frame : {:d}\nCurrent board'.format(t))
    ax = plt.subplot(nrows, ncols+2, (2*(ncols+2)+1, 3*(ncols+2)+2))
    ax.imshow(s[:,:,1], cmap='gray')
    ax.set_title('Frame : {:d}\nPrevious board'.format(t))
    
    # add the convolutional layers (PyTorch: channels first)
    for i in range(nrows):
        for j in range(ncols):
            channel_idx = i*4+j
            if channel_idx < output_temp.shape[0]:
                ax = plt.subplot(nrows, ncols+2, i*(ncols+2) + (j+2) + 1)
                ax.imshow(output_temp[channel_idx, :, :], cmap='gray')
                ax.axis('off')
    
    fig.savefig('images/weight_visual_{:s}_{:02d}.png'.format(version, t), 
                dpi=72, bbox_inches='tight')
    # plt.show()
    # update current state
    s = next_s.copy()
    t += 1

print(f'\nGenerated {t} frame images in images/ folder')

# Try to create video with ffmpeg
import shutil
ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path is None:
    print('\nffmpeg not found in PATH. Video creation skipped.')
    print('Individual frame images saved in images/ folder.')
    print('\nTo create video manually, ensure ffmpeg is in PATH and run:')
    print(f'ffmpeg -y -framerate 1 -pattern_type sequence -i "images/weight_visual_{version}_%02d.png" -c:v libx264 -pix_fmt gray images/weight_visual_{version}_{iteration:04d}_conv{layer_num}.mp4')
else:
    print(f'Creating video with ffmpeg...')
    result = os.system('ffmpeg -y -framerate 1 -pattern_type sequence -i "images/weight_visual_{:s}_%02d.png" \
              -c:v libx264 -pix_fmt gray images/weight_visual_{:s}_{:04d}_conv{:d}.mp4'\
              .format(version, version, iteration, layer_num))
    
    if result == 0:
        print('Video created successfully!')
        # Remove individual frames after successful video creation
        for i in range(t):
            os.remove('images/weight_visual_{:s}_{:02d}.png'.format(version, i))
    else:
        print('Video creation failed. Individual frames kept.')

""" -t 40 specifies pick 40s of the video, fps=1 is 1 frame per second, -loop 0 is
loop till infinity
ffmpeg -t 40 -i images/activation_visual_v15.1_188000_conv1.mp4 -vf "fps=1,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 images\activation_visual_v15.1_188000_conv1.gif -y
"""
