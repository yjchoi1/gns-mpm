# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m gns.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""  # pylint: disable=line-too-long

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string("rollout_path", None, help="Path where rollout pickle file is")
flags.DEFINE_string("rollout_filename", None, help="The name of the rollout to read to render, e.g., rollout_test0-2")
flags.DEFINE_string("output_filename", None, help="The name of output animation to save, e.g., rollout_test0-2")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")
flags.DEFINE_integer("dims", 2, help="dims")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    1: "red", # for droplet
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}


def main(unused_argv):

    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(f"{FLAGS.rollout_path}/{FLAGS.rollout_filename}.pkl", "rb") as file:
        rollout_data = pickle.load(file)

    if FLAGS.dims == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(rollout_data["loss"].cpu().numpy())

        plot_info = []
        for ax_i, (label, rollout_field) in enumerate(
                [("Reality", "ground_truth_rollout"),
                 ("GNS", "predicted_rollout")]):
            # Append the initial positions to get the full trajectory.
            trajectory = np.concatenate([
                rollout_data["initial_positions"],
                rollout_data[rollout_field]], axis=0)
            ax = axes[ax_i]
            ax.set_title(label)
            bounds = rollout_data["metadata"]["bounds"]
            ax.set_xlim(bounds[0][0], bounds[0][1])
            ax.set_ylim(bounds[1][0], bounds[1][1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1.)
            ax.grid(visible=True)
            points = {
                particle_type: ax.plot([], [], "o", ms=1.0, color=color)[0]
                for particle_type, color in TYPE_TO_COLOR.items()}
            plot_info.append((ax, trajectory, points))

        num_steps = trajectory.shape[0]

        def update(step_i):
            outputs = []
            for _, trajectory, points in plot_info:
                for particle_type, line in points.items():
                    mask = rollout_data["particle_types"] == particle_type
                    line.set_data(trajectory[step_i, mask, 0],
                                  trajectory[step_i, mask, 1])
                    outputs.append(line)
            return outputs

        unused_animation = animation.FuncAnimation(
            fig, update,
            frames=np.arange(0, num_steps, FLAGS.step_stride), interval=10)

        unused_animation.save(f'{FLAGS.rollout_path}/{FLAGS.output_filename}.gif', dpi=100, fps=30, writer='imagemagick')
        # plt.show(block=FLAGS.block_on_show)

    elif FLAGS.dims == 3:

        # get rollouts from .pkl file
        trajectory = {}
        rollout_fields = ["ground_truth_rollout", "predicted_rollout"]
        for rollout_field in rollout_fields:
            trajectory[rollout_field] = np.concatenate(
                [rollout_data["initial_positions"], rollout_data[rollout_field]], axis=0
            )
        num_steps = trajectory["ground_truth_rollout"].shape[0]

        # init figures
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        axes = [ax1, ax2]

        # fig creating function
        def animate(i):
            fig.clear()
            xboundary = rollout_data["metadata"]["bounds"][0]
            yboundary = rollout_data["metadata"]["bounds"][1]
            zboundary = rollout_data["metadata"]["bounds"][2]

            rollout_fields = ["ground_truth_rollout", "predicted_rollout"]

            for j, rollout_field in enumerate(rollout_fields):
                axes[j] = fig.add_subplot(1, 2, j+1, projection='3d', autoscale_on=False)
                axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                axes[j].set_zlim([float(zboundary[0]), float(zboundary[1])])
                axes[j].scatter(trajectory[rollout_field][i][:, 0],
                           trajectory[rollout_field][i][:, 1],
                           trajectory[rollout_field][i][:, 2], s=1)
                axes[j].view_init(elev=20., azim=i * 0.5)
                axes[j].grid(True, which='both')

        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, num_steps, 3), interval=10)

        ani.save(f'{FLAGS.rollout_path}/{FLAGS.output_filename}.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {FLAGS.rollout_path}/{FLAGS.output_filename}.gif")



if __name__ == "__main__":
    app.run(main)
