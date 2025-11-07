import os
import json
import torch
import autoroot
import numpy as np
from dataclasses import asdict
from argparse import ArgumentParser
from flask import Flask, jsonify, request
from mmadavla.eval.utils import MMaDA_VLA_Server


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "f79506c83abdf9b03271f4522d854759"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--port", type=int, default=36657)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--prompt_interval_steps", type=int, default=6)
    parser.add_argument("--gen_interval_steps", type=int, default=6)
    parser.add_argument("--transfer_ratio", type=float, default=0.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mmada_vla_server = MMaDA_VLA_Server(
        mmadavla_path=args.mmadavla_path,
        benchmark="calvin",
        device=device,
        temperature=args.temperature,
        timesteps=args.timesteps,
        cache=args.cache,
        prompt_interval_steps=args.prompt_interval_steps,
        gen_interval_steps=args.gen_interval_steps,
        transfer_ratio=args.transfer_ratio,
    )
    flask_app = Flask(__name__)
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            image = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8).reshape((200, 200, 3))
            gripper_image = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8).reshape((84, 84, 3))
            content = json.loads(request.files["json"].read())
            task_inst = content["instruction"]
            images, actions = mmada_vla_server.inference(task_inst=task_inst, image=image, gripper_image=gripper_image)
            return jsonify(actions[-1].tolist())
    flask_app.run(host="0.0.0.0", port=args.port)