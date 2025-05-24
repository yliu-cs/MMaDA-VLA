import os
import copy
import torch
import diffusers
import accelerate
from diffusers import __version__
from accelerate import dispatch_model
from huggingface_hub.utils import validate_hf_hub_args
from typing import Optional, Union, Callable, Dict, Any, Self
from diffusers.models.model_loading_utils import _fetch_index_file
from diffusers.utils import is_accelerate_available, is_torch_version, _add_variant, _get_model_file
from diffusers.models.modeling_utils import no_init_weights, ContextManagers, load_state_dict, _determine_device_map


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SAFETENSORS_WEIGHTS_NAME = "pytorch_model.safetensors"
_LOW_CPU_MEM_USAGE_DEFAULT = True if is_torch_version(">=", "1.9.0") else False


class ModelMixin(diffusers.models.modeling_utils.ModelMixin):
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~models.ModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            max_shard_size (`int` or `str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5GB"`).
                If expressed as an integer, the unit is bytes. Note that this limit will be decreased after a certain
                period of time (starting from Oct 2024) to allow users to upgrade to the latest version of `diffusers`.
                This is to establish a common default size for this argument across different libraries in the Hugging
                Face ecosystem (`transformers`, and `accelerate`, for example).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        hf_quantizer = getattr(self, "hf_quantizer", None)
        if hf_quantizer is not None:
            quantization_serializable = (
                hf_quantizer is not None
                and isinstance(hf_quantizer, DiffusersQuantizer)
                and hf_quantizer.is_serializable
            )
            if not quantization_serializable:
                raise ValueError(
                    f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                    " the logger on the traceback to understand the reason why the quantized model is not serializable."
                )

        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)
        weights_name_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
            ".safetensors", "{suffix}.safetensors"
        )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Save the model
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name_pattern
        )

        # Clean the folder from a previous save
        if is_main_process:
            for filename in os.listdir(save_directory):
                if filename in state_dict_split.filename_to_tensors.keys():
                    continue
                full_filename = os.path.join(save_directory, filename)
                if not os.path.isfile(full_filename):
                    continue
                weights_without_ext = weights_name_pattern.replace(".bin", "").replace(".safetensors", "")
                weights_without_ext = weights_without_ext.replace("{suffix}", "")
                filename_without_ext = filename.replace(".bin", "").replace(".safetensors", "")
                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                if (
                    filename.startswith(weights_without_ext)
                    and _REGEX_SHARD.fullmatch(filename_without_ext) is not None
                ):
                    os.remove(full_filename)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            filepath = os.path.join(save_directory, filename)
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safetensors.torch.save_file(shard, filepath, metadata={"format": "pt"})
            else:
                torch.save(shard, filepath)

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(repo_id, token=token)
            model_card = populate_model_card(model_card)
            model_card.save(Path(save_directory, "README.md").as_posix())

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device. Defaults to `None`, meaning that the model will be loaded on CPU.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        quantization_config = kwargs.pop("quantization_config", None)
        dduf_entries: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_entries", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.float32
            logger.warning(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        # Check if we can handle device_map and dispatching the weights
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            if device_map is not None and not is_torch_version(">=", "1.10"):
                # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
                raise ValueError("`low_cpu_mem_usage` and `device_map` require PyTorch >= 1.10.")

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        unused_kwargs = {}

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            dduf_entries=dduf_entries,
            **kwargs,
        )
        # no in-place modification of the original config.
        config = copy.deepcopy(config)

        # determine initial quantization config.
        #######################################
        pre_quantized = "quantization_config" in config and config["quantization_config"] is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config["quantization_config"] = DiffusersAutoQuantizer.merge_quantization_configs(
                    config["quantization_config"], quantization_config
                )
            else:
                config["quantization_config"] = quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(
                config["quantization_config"], pre_quantized=pre_quantized
            )
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = cls._keep_in_fp32_modules is not None and (
            hf_quantizer is None or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
        )

        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []

        is_sharded = False
        resolved_model_file = None

        # Determine if we're loading from a directory of sharded checkpoints.
        sharded_metadata = None
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file_kwargs = {
            "is_local": is_local,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder or "",
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "variant": variant,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
            "dduf_entries": dduf_entries,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (index_file is None or not os.path.exists(index_file)):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)
        if index_file is not None and (dduf_entries or index_file.is_file()):
            is_sharded = True

        if is_sharded and from_flax:
            raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

        # load model
        if from_flax:
            resolved_model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights
            from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model

            model = load_flax_checkpoint_in_pytorch_model(model, resolved_model_file)
        else:
            # in the case it is sharded, we have already the index
            if is_sharded:
                resolved_model_file, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                    dduf_entries=dduf_entries,
                )
            elif use_safetensors:
                try:
                    resolved_model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                        dduf_entries=dduf_entries,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if resolved_model_file is None and not is_sharded:
                resolved_model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                    dduf_entries=dduf_entries,
                )

        if not isinstance(resolved_model_file, list):
            resolved_model_file = [resolved_model_file]

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is float8, we don't use _set_default_torch_dtype and we downcast after loading the model
        dtype_orig = None
        if torch_dtype is not None and not torch_dtype == getattr(torch, "float8_e4m3fn", None):
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(
                    f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
                )
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        init_contexts = [no_init_weights()]

        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            model = cls.from_config(config, **unused_kwargs)

        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        state_dict = None
        if not is_sharded:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_model_file[0], disable_mmap=disable_mmap, dduf_entries=dduf_entries)
            # We only fix it for non sharded checkpoints as we don't need it yet for sharded one.
            model._fix_state_dict_keys_on_load(state_dict)

        if is_sharded:
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_keys = list(state_dict.keys())

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

        # Now that the model is loaded, we can determine the device_map
        device_map = _determine_device_map(
            model, device_map, max_memory, torch_dtype, keep_in_fp32_modules, hf_quantizer
        )
        if hf_quantizer is not None:
            hf_quantizer.validate_environment(device_map=device_map)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            resolved_model_file,
            pretrained_model_name_or_path,
            loaded_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            hf_quantizer=hf_quantizer,
            keep_in_fp32_modules=keep_in_fp32_modules,
            dduf_entries=dduf_entries,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        # Dispatch model with hooks on all devices if necessary
        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
            }
            dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if (
            torch_dtype is not None
            and torch_dtype == getattr(torch, "float8_e4m3fn", None)
            and hf_quantizer is None
            and not use_keep_in_fp32_modules
        ):
            model = model.to(torch_dtype)

        if hf_quantizer is not None:
            # We also make sure to purge `_pre_quantization_dtype` when we serialize
            # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
            model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
        else:
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            return model, loading_info

        return model