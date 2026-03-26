import os
import json
import torch
import autoroot
import numpy as np
from transformers import set_seed
from argparse import ArgumentParser
from flask import Flask, jsonify, request
from mmadavla.eval.utils import MMaDA_VLA_Server


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "CALVIN", "ABC_D", "lr1e-4"))
    parser.add_argument("--mmadavla_path", type=str, default=os.path.join(os.getcwd(), "ckpt", "MMaDA-VLA", "CALVIN", "ABC_D", "chunk10_wo_pad", "lr1e-4"))
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--port", type=int, default=36657)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    mmadavla_server = MMaDA_VLA_Server(mmadavla_path=args.mmadavla_path, benchmark="calvin", device=device)
    flask_app = Flask(__name__)
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            try:
                batch_task_insts = json.loads(request.files["json"].read())["batch_task_insts"]
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                exit()
            batch_images = np.frombuffer(request.files["batch_images"].read(), dtype=np.uint8).reshape((len(batch_task_insts), 200, 200, 3))
            batch_gripper_images = np.frombuffer(request.files["batch_gripper_images"].read(), dtype=np.uint8).reshape((len(batch_task_insts), 84, 84, 3))
            batch_actions = mmadavla_server.batch_inference(
                task_insts=batch_task_insts,
                images=batch_images,
                gripper_images=batch_gripper_images,
            )
            return jsonify(batch_actions.tolist())
        else:
            raise NotImplementedError(f"Method {request.method} not implemented")
    flask_app.run(host="0.0.0.0", port=args.port)