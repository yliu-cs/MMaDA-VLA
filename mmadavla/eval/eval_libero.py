import os
import re
import time
import math
import json
import torch
import imageio
import autoroot
import numpy as np
from PIL import Image
from enum import Enum
from tqdm.auto import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from collections import deque, OrderedDict
from typing import List, Dict, Tuple, Union
from argparse import ArgumentParser, Namespace
from mmadavla.eval.utils import MMaDA_VLA_Server
from libero.libero.envs import OffScreenRenderEnv


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--task_suite", type=TaskSuite, default=TaskSuite.LIBERO_OBJECT, choices=list(TaskSuite))
    parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "29c912d2043a232ab44523d4c69c2d70", "checkpoint_10"))
    parser.add_argument("--initial_states_path", type=str, default="DEFAULT")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--resize_size", type=int, default=256)
    args = parser.parse_args()
    return args


def get_libero_env(task: benchmark.Task, resolution: int = 256) -> Tuple[OffScreenRenderEnv, str]:
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action() -> List[int]:
    return [0, 0, 0, 0, 0, 0, -1]


def load_initial_states(
    args: Namespace,
    task_suite: benchmark.LIBERO_OBJECT,
    task_id: int,
) -> Tuple[np.ndarray, Union[None, Dict]]:
    initial_states = task_suite.get_task_init_states(task_id)
    if args.initial_states_path != "DEFAULT":
        with open(args.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        return initial_states, all_initial_states
    else:
        return initial_states, None


def save_rollout_video(rollout_images: List[np.ndarray], idx: int, success: bool, task_description: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(save_dir, f"{time.strftime('%Y_%m_%d-%H_%M_%S')}--mmadavla--episode={idx}--success={success}--task={processed_task_description}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat[3] = max(min(quat[3], 1.0), -1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def prepare_observation(obs: OrderedDict, resize_size: int) -> Tuple[Dict, np.ndarray]:
    img, wrist_img = obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
    }
    return observation, img


def run_episodes(
    args: Namespace,
    env: OffScreenRenderEnv,
    resize_size: int,
    task_description: str,
    mmada_vla: MMaDA_VLA_Server,
    initial_state: np.ndarray = None
) -> Tuple[bool, List[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    # if args.num_open_loop_steps != action_chunk_size:
    #     raise NotImplementedError
    action_queue = deque(maxlen=args.num_open_loop_steps)
    t, replay_images, max_steps, success = 0, [], TASK_MAX_STEPS[args.task_suite], False
    try:
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action())
                t += 1
                continue
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            if len(action_queue) == 0:
                images, actions = mmada_vla.inference(
                    task_inst=f"{task_description}\n{' '.join(map(str, observation['state'].flatten()))}",
                    image=observation["full_image"],
                    gripper_image=observation["wrist_image"],
                )
                actions = actions.tolist()
                action_queue.extend(actions)
            action = action_queue.popleft()
            obs, reward, done, info = env.step(action)
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        print(f"Episode error: {e}")
        import traceback
        traceback.print_exc()
        exit()
    return success, replay_images


def run_task(
    args: Namespace,
    task_suite: benchmark.LIBERO_OBJECT,
    task_id: int,
    resize_size: int,
    mmada_vla: MMaDA_VLA_Server,
    total_episodes: int = 0,
    total_successes: int = 0,
) -> Tuple[int, int]:
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(args=args, task_suite=task_suite, task_id=task_id)
    env, task_description = get_libero_env(task=task)
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm(range(args.num_trials_per_task)):
        print(f"Task: {task_description}")
        if args.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                print(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!")
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
        print(f"Starting episode {task_episodes + 1} ...")
        success, replay_images = run_episodes(
            args=args,
            env=env,
            resize_size=resize_size,
            task_description=task_description,
            mmada_vla=mmada_vla,
            initial_state=initial_state,
        )
        task_episodes, total_episodes = task_episodes + 1, total_episodes + 1
        task_successes, total_successes = task_successes + (1 if success else 0), total_successes + (1 if success else 0)
        if not success:
            save_rollout_video(
                rollout_images=replay_images,
                idx=total_episodes,
                success=success,
                task_description=task_description,
                save_dir=os.path.join(os.getcwd(), "rollouts", (re.search(r"(.*/)?(MMaDA-VLA/.*)", args.mmadavla_path)).group(2).split("/", 1)[1])
            )
        print(f"Success: {success}  # episodes completed so far: {total_episodes} # successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    print(f"Current task success rate: {task_success_rate}, Current total success rate: {total_success_rate}")
    return total_episodes, total_successes


def main(args: Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mmada_vla = MMaDA_VLA_Server(mmadavla_path=args.mmadavla_path, benchmark="libero", device=device)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks):
        total_episodes, total_successes = run_task(
            args=args,
            task_suite=task_suite,
            task_id=task_id,
            resize_size=args.resize_size,
            mmada_vla=mmada_vla,
            total_episodes=total_episodes,
            total_successes=total_successes,
        )
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print(f"{total_episodes=} {total_successes} ({final_success_rate * 100:.1f}%)")
    return final_success_rate


if __name__ == "__main__":
    args = get_args()
    main(args=args)