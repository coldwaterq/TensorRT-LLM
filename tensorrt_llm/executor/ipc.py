import json
import time
import traceback
from queue import Queue
from typing import Any, Dict, Optional

import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger
def get_postproc_params():
    from .postproc_worker import PostprocParams
    return PostprocParams
def get_postproc_args():
    from .postproc_worker import PostprocArgs
    return PostprocArgs

# Import GenerationRequest lazily to avoid circular imports
def get_generation_request():
    from .request import GenerationRequest
    return GenerationRequest

def get_cancelling_request():
    from .request import CancellingRequest
    return CancellingRequest

from ..llmapi.utils import (ManagedThread, enable_llm_debug, nvtx_mark,
                            nvtx_range, print_colored, print_colored_debug)
from ..sampling_params import SamplingParams
from ..bindings.executor import Response

# Import postprocessors lazily to avoid circular imports
def get_postprocessor(name: str):
    from tensorrt_llm.serve.postprocess_handlers import (completion_response_post_processor,
                                                       completion_stream_post_processor)
    return {
        "completion_response_post_processor": completion_response_post_processor,
        "completion_stream_post_processor": completion_stream_post_processor,
    }[name]

class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
    }

    # Base serializable types and their handlers
    BASE_SERIALIZABLE_TYPES = {
        "bytes": {
            "check": lambda obj: isinstance(obj, bytes),
            "serialize": lambda obj: {"__serial_type__": "bytes", "data": obj.hex()},
            "deserialize": lambda obj: bytes.fromhex(obj["data"])
        },
        "Exception": {
            "check": lambda obj: isinstance(obj, Exception),
            "serialize": lambda obj: {
                "__serial_type__": "Exception",
                "type": obj.__class__.__name__,
                "message": str(obj),
                "traceback": traceback.format_exc()
            },
            "deserialize": lambda obj: type(obj["type"], (Exception,), {})(obj["message"])
        },
        "GenerationRequest": {
            "check": lambda obj: obj.__class__.__name__ == "GenerationRequest",
            "serialize": lambda obj: {
                "__serial_type__": "GenerationRequest",
                "prompt_token_ids": obj.prompt_token_ids,
                "query_token_ids": obj.query_token_ids,
                "sampling_params": ZeroMqQueue.BASE_SERIALIZABLE_TYPES["SamplingParams"]["serialize"](obj.sampling_params),
                "postproc_params": ZeroMqQueue.BASE_SERIALIZABLE_TYPES["PostprocParams"]["serialize"](obj.postproc_params) if obj.postproc_params else None,
                "lora_request": obj.lora_request,
                "prompt_adapter_request": obj.prompt_adapter_request,
                "streaming": obj.streaming,
                "prompt_tuning_config": obj.prompt_tuning_config,
                "mrope_config": obj.mrope_config,
                "kv_cache_retention_config": obj.kv_cache_retention_config,
                "id": obj.id,
                "disaggregated_params": obj.disaggregated_params
            },
            "deserialize": lambda obj: get_generation_request()(
                prompt_token_ids=obj["prompt_token_ids"],
                query_token_ids=obj["query_token_ids"],
                sampling_params=ZeroMqQueue.BASE_SERIALIZABLE_TYPES["SamplingParams"]["deserialize"](obj["sampling_params"]),
                postproc_params=ZeroMqQueue.BASE_SERIALIZABLE_TYPES["PostprocParams"]["deserialize"](obj["postproc_params"]) if obj["postproc_params"] else None,
                lora_request=obj["lora_request"],
                prompt_adapter_request=obj["prompt_adapter_request"],
                streaming=obj["streaming"],
                prompt_tuning_config=obj["prompt_tuning_config"],
                mrope_config=obj["mrope_config"],
                kv_cache_retention_config=obj["kv_cache_retention_config"],
                disaggregated_params=obj["disaggregated_params"]
            ).set_id(obj["id"])
        },
        "CancellingRequest": {
            "check": lambda obj: obj.__class__.__name__ == "CancellingRequest",
            "serialize": lambda obj: {
                "__serial_type__": "CancellingRequest"
            },
            "deserialize": lambda obj: get_cancelling_request()()
        },
        "SamplingParams": {
            "check": lambda obj: isinstance(obj, SamplingParams),
            "serialize": lambda obj: {
                "__serial_type__": "SamplingParams",
                "end_id": obj.end_id,
                "pad_id": obj.pad_id,
                "max_tokens": obj.max_tokens,
                "max_new_tokens": obj.max_new_tokens,
                "bad": obj.bad,
                "bad_token_ids": obj.bad_token_ids,
                "temperature": obj.temperature,
                "top_k": obj.top_k,
                "top_p": obj.top_p,
                "top_p_min": obj.top_p_min,
                "top_p_reset_ids": obj.top_p_reset_ids,
                "top_p_decay": obj.top_p_decay,
                "seed": obj.seed,
                "random_seed": obj.random_seed,
                "length_penalty": obj.length_penalty,
                "early_stopping": obj.early_stopping,
                "repetition_penalty": obj.repetition_penalty,
                "min_length": obj.min_length,
                "presence_penalty": obj.presence_penalty,
                "frequency_penalty": obj.frequency_penalty,
                "no_repeat_ngram_size": obj.no_repeat_ngram_size,
                "min_p": obj.min_p,
                "beam_width_array": obj.beam_width_array,
                "return_log_probs": obj.return_log_probs,
                "return_context_logits": obj.return_context_logits,
                "return_generation_logits": obj.return_generation_logits
            },
            "deserialize": lambda obj: SamplingParams(
                end_id=obj["end_id"],
                pad_id=obj["pad_id"],
                max_tokens=obj["max_tokens"],
                max_new_tokens=obj["max_new_tokens"],
                bad=obj["bad"],
                bad_token_ids=obj["bad_token_ids"],
                temperature=obj["temperature"],
                top_k=obj["top_k"],
                top_p=obj["top_p"],
                top_p_min=obj["top_p_min"],
                top_p_reset_ids=obj["top_p_reset_ids"],
                top_p_decay=obj["top_p_decay"],
                seed=obj["seed"],
                random_seed=obj["random_seed"],
                length_penalty=obj["length_penalty"],
                early_stopping=obj["early_stopping"],
                repetition_penalty=obj["repetition_penalty"],
                min_length=obj["min_length"],
                presence_penalty=obj["presence_penalty"],
                frequency_penalty=obj["frequency_penalty"],
                no_repeat_ngram_size=obj["no_repeat_ngram_size"],
                min_p=obj["min_p"],
                beam_width_array=obj["beam_width_array"],
                return_log_probs=obj["return_log_probs"],
                return_context_logits=obj["return_context_logits"],
                return_generation_logits=obj["return_generation_logits"]
            )
        },
        "PostprocParams": {
            "check": lambda obj: obj.__class__.__name__ == "PostprocParams",
            "serialize": lambda obj: {
                "__serial_type__": "PostprocParams",
                "post_processor": {
                    "__serial_type__": "PostProcessorFunction",
                    "name": obj.post_processor.__name__ if obj.post_processor else None
                },
                "postproc_args": {
                    "first_iteration": obj.postproc_args.first_iteration,
                    "num_prompt_tokens": obj.postproc_args.num_prompt_tokens,
                    "tokenizer": obj.postproc_args.tokenizer
                }
            },
            "deserialize": lambda obj: get_postproc_params()(
                post_processor=get_postprocessor(obj["post_processor"]["name"]) if obj["post_processor"]["name"] else None,
                postproc_args=get_postproc_args()(
                    first_iteration=obj["postproc_args"]["first_iteration"],
                    num_prompt_tokens=obj["postproc_args"]["num_prompt_tokens"],
                    tokenizer=obj["postproc_args"]["tokenizer"]
                )
            )
        },
        "Result": {
            "check": lambda obj: obj.__class__.__name__ == "Result",
            "serialize": lambda obj: {
                "__serial_type__": "Result",
                "additional_outputs": obj.additional_outputs,
                "context_logits": obj.context_logits,
                "context_phase_params": obj.context_phase_params,
                "cum_log_probs": obj.cum_log_probs,
                "decoding_iter": obj.decoding_iter,
                "encoder_output": obj.encoder_output,
                "finish_reasons": [reason.value for reason in obj.finish_reasons],
                "generation_logits": obj.generation_logits,
                "is_final": obj.is_final,
                "is_sequence_final": obj.is_sequence_final,
                "log_probs": obj.log_probs,
                "output_token_ids": obj.output_token_ids,
                "request_perf_metrics": obj.request_perf_metrics,
                "sequence_index": obj.sequence_index,
                "spec_dec_fast_logits_info": obj.spec_dec_fast_logits_info
            },
            "deserialize": lambda obj: Result(
                additional_outputs=obj["additional_outputs"],
                context_logits=obj["context_logits"], 
                context_phase_params=obj["context_phase_params"],
                cum_log_probs=obj["cum_log_probs"],
                decoding_iter=obj["decoding_iter"],
                encoder_output=obj["encoder_output"],
                finish_reasons=[FinishReason(reason) for reason in obj["finish_reasons"]],
                generation_logits=obj["generation_logits"],
                is_final=obj["is_final"],
                is_sequence_final=obj["is_sequence_final"], 
                log_probs=obj["log_probs"],
                output_token_ids=obj["output_token_ids"],
                request_perf_metrics=obj["request_perf_metrics"],
                sequence_index=obj["sequence_index"],
                spec_dec_fast_logits_info=obj["spec_dec_fast_logits_info"]
            )
        },
        "Response": {
            "check": lambda obj: obj.__class__.__name__ == "Response",
            "serialize": lambda obj: {
                "__serial_type__": "Response",
                "request_id": obj.request_id,
                "client_id": obj.client_id,
                "error_msg": obj.error_msg if obj.has_error() else "",
                "result": ZeroMqQueue.BASE_SERIALIZABLE_TYPES["Result"]["serialize"](obj.result)
            },
            "deserialize": lambda obj: Response(
                request_id=obj["request_id"],
                error_msg=obj["error_msg"],
                client_id=obj["client_id"],
                result=ZeroMqQueue.BASE_SERIALIZABLE_TYPES["Result"]["deserialize"](obj["result"])
            )
        }
    }

    def __init__(self,
                 address: Optional[str] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 additional_serializable_types: Optional[Dict] = None):
        '''
        Parameters:
            address (Tuple[str, str], optional): The address (tcp-ip_port, authkey) for the IPC. Defaults to None.
            is_server (bool): Whether the current process is the server or the client.
            additional_serializable_types (Dict, optional): Additional types to be added to the serializable types.
        '''

        self.socket_type = socket_type
        self.address = address or "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket_type = socket_type

        # Initialize SERIALIZABLE_TYPES with base types
        self.SERIALIZABLE_TYPES = self.BASE_SERIALIZABLE_TYPES.copy()
        
        # Add any additional serializable types
        if additional_serializable_types:
            self.SERIALIZABLE_TYPES.update(additional_serializable_types)

        self.socket = self.context.socket(socket_type)

        if (socket_type == zmq.PAIR
                and self.is_server) or socket_type == zmq.PULL:
            self.socket.bind(
                self.address
            )  # Binds to the address and occupy a port immediately
            self.address = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            print_colored_debug(
                f"Server [{name}] bound to {self.address} in {self.socket_type_str[socket_type]}\n",
                "green")

    def setup_lazily(self):
        if self._setup_done:
            return
        self._setup_done = True

        if not self.is_server:
            print_colored_debug(
                f"Client [{self.name}] connecting to {self.address} in {self.socket_type_str[self.socket_type]}\n",
                "green")
            self.socket.connect(self.address)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def _serialize_obj(self, obj: Any) -> str:
        """Helper method to safely serialize objects to JSON using approved types"""
        for type_handler in self.SERIALIZABLE_TYPES.values():
            if type_handler["check"](obj):
                serialized_obj = type_handler["serialize"](obj)
                try:
                    return json.dumps(serialized_obj)
                except Exception as e:
                    logger.error(f"Error serializing object: {e}")
                    logger.error(f"Serialized object: {serialized_obj}")
                    logger.error(traceback.format_exc())
                    raise e
                
        # If object is not in approved list, try basic JSON serialization
        try:
            return json.dumps(obj)
        except TypeError as e:
            if isinstance(obj, list):
                serialized_objs = {"__serial_type__": "ObjectList", "objs": []}
                for o in obj:
                    serialized_objs["objs"].append(self._serialize_obj(o))
                return json.dumps(serialized_objs)
            raise TypeError(f"Object {obj} is not in approved serializable types") from e

    def _deserialize_obj(self, data: str) -> Any:
        """Helper method to deserialize objects from JSON using approved types"""
        obj = json.loads(data)
        if isinstance(obj, dict):
            if "__serial_type__" in obj.keys():
                if obj["__serial_type__"] == "ObjectList":
                    return [self._deserialize_obj(o) for o in obj["objs"]]
                return self.SERIALIZABLE_TYPES[obj["__serial_type__"]]["deserialize"](obj)
        return obj

    def put(self, obj: Any):
        self.setup_lazily()
        with nvtx_range("send", color="blue", category="IPC"):
            self.socket.send_string(self._serialize_obj(obj))

    async def put_async(self, obj: Any):
        self.setup_lazily()
        try:
            await self.socket.send_string(self._serialize_obj(obj))
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

        nvtx_mark("ipc.send", color="blue", category="IPC")

    def get(self) -> Any:
        self.setup_lazily()
        return self._deserialize_obj(self.socket.recv_string())

    async def get_async(self) -> Any:
        self.setup_lazily()
        return self._deserialize_obj(await self.socket.recv_string())

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def __del__(self):
        self.close()


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[str] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(obj)
        else:
            batch = obj if isinstance(obj, list) else [obj]
            self.queue.put(batch)

    def get(self) -> Any:
        return self.queue.get()

    @property
    def address(self) -> str:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
