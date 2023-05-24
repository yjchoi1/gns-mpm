# Refer to https://github.com/echowve/meshGraphNets_pytorch/tree/master

import sys
import os
import glob
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_geometric.transforms as T
import re
import pickle
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append('/work2/08264/baagee/frontera/gns-mpm-dev/gns-meshnet')
from gns import mesh_data_loader
from gns import distribute
from gns import mesh_simulator
from utils.noise import get_velocity_noise
from utils.utils import datas_to_graph
from utils.utils import NodeType
from utils.utils import optimizer_to


INPUT_SEQUENCE_LENGTH = 1
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

# data_path = "/work2/08264/baagee/frontera/meshnet/data/cylinder_flow_npz/"
# model_path = "/work2/08264/baagee/frontera/meshnet/save_models/cylinder_flow_npz/"
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_string('data_path', "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/pipe-npz/", help='The dataset directory.')
flags.DEFINE_string('model_path', "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/models/pipe-npz/", help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/rollouts/pipe-npz/", help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', None, help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')

FLAGS = flags.FLAGS


batch_size = 1  # TODO: change batch_size when actually do training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
noise_std = 2e-2
node_type_embedding_size = 9
dt = 0.01

lr_init = 1e-4
lr_decay = 1.0
lr_decay_steps = 5e6
ntraining_steps = 1e6
nsave_steps = 2000
print_steps = 10


def predict(simulator: mesh_simulator.MeshSimulator,
            device: str):

    # Load simulator
    if os.path.exists(FLAGS.model_path + FLAGS.model_file):
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    else:
        raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    # load trajectory data.
    ds = mesh_data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

    # eval_loss = []
    with torch.no_grad():
        for i, features in enumerate(ds):
            nsteps = len(features[0]) - INPUT_SEQUENCE_LENGTH
            prediction_data = rollout(simulator, features, nsteps, device)
            print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']}")
            # eval_loss.append(prediction_data['mean_loss'])

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                filename = f'{FLAGS.rollout_filename}_{i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(prediction_data, f)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']}")
    a = 1

def rollout(simulator: mesh_simulator.MeshSimulator,
            features,
            nsteps: int,
            device):

    node_coords = features[0]  # (timesteps, nnode, ndims)
    node_types = features[1]  # (timesteps, nnode, )
    velocities = features[2]  # (timesteps, nnode, ndims)
    pressures = features[3]  # (timesteps, nnode, )
    cells = features[4]  # # (timesteps, ncells, nnode_per_cell)

    initial_velocities = velocities[:INPUT_SEQUENCE_LENGTH]
    ground_truth_velocities = velocities[INPUT_SEQUENCE_LENGTH:]

    current_velocities = initial_velocities.squeeze().to(device)
    predictions = []
    mask = None

    for step in tqdm(range(nsteps), total=nsteps):
        # predict next velocity
        # print(f"Rollout step: {step}/{nsteps}")

        # obtain data to form a graph
        current_node_coords = node_coords[step]
        current_node_type = node_types[step]
        current_pressure = pressures[step]
        current_cell = cells[step]
        current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        next_ground_truth_velocities = ground_truth_velocities[step].to(device)
        current_example = (
            (current_node_coords, current_node_type, current_velocities, current_pressure, current_cell, current_time_idx_vector),
            next_ground_truth_velocities)

        # make graph
        graph = datas_to_graph(current_example, dt=dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph)


        predicted_next_velocity = simulator.predict_velocity(
            current_velocities=graph.x[:, 1:3],
            node_type=graph.x[:, 0],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr)

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the rest of timesteps
            mask = torch.logical_or(current_node_type == NodeType.NORMAL, current_node_type == NodeType.OUTFLOW)
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1)
            # Maintain previous velocity if node_type is not (Normal or Outflow).
            # i.e., only update normal or outflow nodes.
            predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
        predictions.append(predicted_next_velocity)

        # Update current position for the next prediction
        current_velocities = predicted_next_velocity.to(device)

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    loss = (predictions - ground_truth_velocities.to(device)) ** 2

    output_dict = {
        'initial_velocities': initial_velocities.cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_velocities.cpu().numpy(),
        'node_coords': node_coords.cpu().numpy(),
        'node_types': node_types.cpu().numpy(),
        'mean_loss': loss.mean().cpu().numpy()
    }

    return output_dict



def train(simulator):

    print(f"device = {device}")

    ## INITIATE TRAINING
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)
    step = 0

    ## SET MODEL AND SAVE PATH & LOAD MODEL
    # If model_path does not exist create new directory and begin training.
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:

        if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{model_path}*model*pt")
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            FLAGS.model_file = f"model-{max_model_number}.pt"
            FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
            # load model
            simulator.load(model_path + FLAGS.model_file)

            # load train state
            train_state = torch.load(model_path + FLAGS.train_state_file)
            # set optimizer state
            optimizer = torch.optim.Adam(simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device)
            # set global train state
            step = train_state["global_train_state"].pop("step")
        else:
            raise FileNotFoundError(
                f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found.")


    simulator.train()
    simulator.to(device)

    # LOAD DATASET, TODO: change `.npz` name
    ds = mesh_data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/test.npz',
                                                     input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                     batch_size=batch_size)
    not_reached_nsteps = True

    try:
        while not_reached_nsteps:
            for i, example in enumerate(ds):
                # (positions, node_type, velocity_feature, pressure, cells, time_idx_vector, n_node_per_example),
                # velocity_target)

                # make graph
                graph = datas_to_graph(example, dt=dt, device=device)
                # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                graph = transformer(graph)

                # get inputs
                node_types = graph.x[:, 0]
                current_velocities = graph.x[:, 1:3]
                edge_index = graph.edge_index
                edge_features = graph.edge_attr
                target_velocities = graph.y

                velocity_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)

                pred_acc, target_acc = simulator.predict_acceleration(
                    current_velocities=current_velocities,
                    node_type=node_types,
                    edge_index=edge_index,
                    edge_features=edge_features,
                    target_velocities=target_velocities,
                    velocity_noise=velocity_noise)

                # compute loss
                mask = torch.logical_or(node_types == NodeType.NORMAL, node_types == NodeType.OUTFLOW)
                errors = ((pred_acc - target_acc)**2)[mask]
                loss = torch.mean(errors)

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update learning rate
                lr_new = lr_init * (lr_decay ** (step / lr_decay_steps))
                for param in optimizer.param_groups:
                    param['lr'] = lr_new

                if step % print_steps == 0:
                    print(f"Training step: {step}/{ntraining_steps}. Loss: {loss}.")

                # Save model state
                if step % nsave_steps == 0:
                    simulator.save(model_path + 'model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                    torch.save(train_state, f"{model_path}train_state-{step}.pt")

                # Complete training
                if (step >= ntraining_steps):
                    not_reached_nsteps = False
                    break

                step += 1

    except KeyboardInterrupt:
        pass


def main(_):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')

    # load simulator
    simulator = mesh_simulator.MeshSimulator(
        simulation_dimensions=2,
        nnode_in=11,
        nedge_in=3,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        nnode_types=3,
        node_type_embedding_size=9,
        device=device)
    if FLAGS.mode == 'train':
        train(simulator)
    elif FLAGS.mode in ['valid', 'rollout']:
        predict(simulator, device)

if __name__ == "__main__":
    app.run(main)