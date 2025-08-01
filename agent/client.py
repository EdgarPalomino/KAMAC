import datetime
import httpx
import openai
import anthropic
import os
from openai import OpenAI
import time
from functools import wraps
import logging
import sys
from tenacity import retry, stop_after_attempt, wait_exponential
from subprocess import Popen, TimeoutExpired
import threading
import base64
import io
import numpy as np
from typing import Literal
import json
import copy
from filelock import FileLock
sys.path.append(os.path.dirname(__file__))
from logging_utils import print_log, create_logger
import subprocess
from concurrent.futures import ThreadPoolExecutor
import ollama
from ollama import ChatResponse, chat
from PIL import Image
import shutil

base_logger = logging.getLogger(os.path.basename(__file__))


class CachedBase:

    def init(self, saved_fname):
        self.work_dir = os.environ["work_dir"]
        self.cache_path = os.environ["cache_path"]
        self.cached_mode = os.environ["cached_mode"]
        self.cached_action_keys = os.getenv("cached_action_keys", "").split(",")
        self.saved_fname = saved_fname
        self.saved_as_path = os.environ["saved_as_path"]
        self.saved_messages_new = {self.saved_fname: {}}

        if not hasattr(self, "saved_messages") and self.cached_mode != "skip":
            file_path = f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.json"
            if os.path.exists(file_path):
                if os.path.getsize(file_path) > 0:
                    try:
                        json_file = open(file_path, "r", encoding="utf-8")
                        text = json_file.read()
                        self.saved_messages = json.loads(text)
                    except UnicodeDecodeError:
                        json_file.close()
                        with open(file_path, "r", encoding="gbk") as json_file:
                            text = json_file.read()
                        self.saved_messages = json.loads(text)
                        with open(file_path, "w", encoding="utf-8") as json_file:
                            json.dump(
                                self.saved_messages,
                                json_file,
                                ensure_ascii=False,
                                indent=4,
                            )

                    if (
                        "all" not in self.cached_action_keys
                        and self.cached_mode != "all"
                    ):
                        saved_messages = {self.saved_fname: {}}
                        if "initial_assessment-" in self.cached_action_keys:
                            for action, value in self.saved_messages[
                                self.saved_fname
                            ].items():

                                if "recruit-recruiter" in action:
                                    saved_messages[self.saved_fname].update(
                                        {action: value}
                                    )
                                if "initial_assessment-" in action:
                                    saved_messages[self.saved_fname].update(
                                        {action: value}
                                    )
                                if "examplers-" in action:
                                    saved_messages[self.saved_fname].update(
                                        {action: value}
                                    )

                        else:
                            saved_messages[self.saved_fname].update(
                                {
                                    action: value
                                    for action, value in self.saved_messages[
                                        self.saved_fname
                                    ].items()
                                    if action in self.cached_action_keys
                                }
                            )
                        self.saved_messages = saved_messages

                else:
                    self.saved_messages = {self.saved_fname: {}}

            else:
                with FileLock(
                    f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.lock"
                ):
                    with open(
                        f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.json",
                        "w",
                    ) as json_file:
                        self.saved_messages = {self.saved_fname: {}}

    def load_saved_messages(self, action, role, use_cached):
        action = f"{action}-{role}"
        if self.cached_mode != "skip":
            if action in self.saved_messages[self.saved_fname].keys() and (
                self.cached_mode == "all" or use_cached
            ):
                return self.saved_messages[self.saved_fname][action]
            else:
                if not (self.cached_mode == "all" or use_cached):
                    print(
                        f"For {action} use cached results is forbidden. We will update local cache."
                    )
                else:
                    print(f"{action} isn't saved. We will update local cache.")

        return None

    def do_save_messages(self, action, saved_messages, role, use_cached):
        action = f"{action}-{role}" if role != "" else f"{action}"

        if action in self.saved_messages[self.saved_fname].keys():
            self.saved_messages_new[self.saved_fname][action] = saved_messages
            self.saveas_json()
        else:
            self.saved_messages[self.saved_fname][action] = saved_messages
            self.saveas_json()

            # shutil.move(
            #     f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.json",
            #     f"{self.work_dir}/{self.cache_path}/{self.saved_fname}_1.json",
            # )

    def saveas_json(self):
        with FileLock(f"{self.work_dir}/{self.saved_as_path}/{self.saved_fname}.lock"):
            with open(
                f"{self.work_dir}/{self.saved_as_path}/{self.saved_fname}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    (
                        self.saved_messages
                        if self.saved_as_path == self.cache_path
                        else self.saved_messages_new
                    ),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def save_round_comment(self, action, comment, role, use_cached):
        num_agents = len(comment)
        comment = [f"{k}:{v}" for k, v in comment.items()]
        comment = "|".join(comment)
        self.saved_messages[self.saved_fname][f"{action}-role-comment"] = comment
        self.saved_messages[self.saved_fname][f"{action}-num_agents"] = num_agents
        with FileLock(f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.lock"):
            with open(
                f"{self.work_dir}/{self.cache_path}/{self.saved_fname}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.saved_messages,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )


def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time() 
        duration = end_time - start_time 
        base_logger.info(
            f"Function '{func.__name__}' took {duration:.4f} seconds to execute."
        )
        return result

    return wrapper


def retry_on_api_error(func):
    max_retries = os.environ.get("max_retries", 5)
    retry_delay = os.environ.get("retry_delay", 1)

    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                response = func(*args, **kwargs)
                if response is None:
                    continue
                else:
                    return response
            except openai.APIError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: 
                    time.sleep(retry_delay)
                else:
                    print(
                        "All attempts failed. Please check the API status or your request."
                    )
                    return "nan"
            except TimeoutError as e:
                print(f"retrying {attempt}")
                continue
            except KeyboardInterrupt as e:
                raise KeyboardInterrupt
        return "nan"

    return wrapper


def get_openai_client(api_key, base_url=None):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_claude_client(api_key):
    return anthropic.Anthropic(api_key=api_key)


def get_deepseek_client(api_key, base_url="https://api.deepseek.com"):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_gemini_client(
    api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_ollama_client(base_url="http://localhost:11434/v1/"):
    return openai.OpenAI(base_url=base_url)


def get_vllm_client(base_url="http://localhost:8000/v1/"):
    return openai.OpenAI(base_url=base_url)


class Agent_ollama:

    def __init__(
        self,
        instruction,
        role,
        model_info=None,
        source=None,
        examplers=None,
        img_path=None,
        logger=None,
    ):
        self.instruction = instruction
        self.role = role
        self.model_info = (
            model_info if model_info is not None else os.environ["MODEL_INFO"]
        )
        # self.source = source if source is not None else os.environ["SOURCE"]
        self.img_path = img_path
        self.logger = logger
        self.content = ""

    # @retry(stop=stop_after_attempt(3))
    # def chat(self, message, img_path=None, chat_mode=True):
    #     results = subprocess.run(
    #         [
    #             "ollama",
    #             "run",
    #             self.model_info,
    #             f"{message})",
    #         ],
    #         capture_output=True,
    #         text=True,
    #         timeout=600,
    #     )
    #     self.logger.info(f"{results.stdout}")
    #     return results.stdout

    def _chat(self, message, img_path=None, timeout=300):
        try:
            process = subprocess.Popen(
                [
                    "ollama",
                    "run",
                    self.model_info,
                    f"{message})",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            content = ""
            while True:
                output = process.stdout.read(1)
                if output == "" and process.poll() is not None:
                    break
                if output:
                    content += output
                    print(f"{output}", end="")
        except Exception as e:
            print("_chat:", e)
            return None, -1

        return content, process.returncode

    def remaining_time(self, endtime):
        if endtime is None:
            return None
        else:
            return endtime - time.time()

    @retry_on_api_error
    def chat(self, message, img_path=None, debug=False, timeout=300):

        if debug:
            content = str(
                {
                    "trial": [
                        {"lr": 0.001, "k2": 0.01},
                        {"lr": 0.0005, "k2": 0.0005},
                    ]
                }
            )
            content = "```yaml\n" + content + "```\n"
            return content

        #### it can't control the timeout although it is more convenient
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     future = executor.submit(self._chat, message, img_path, timeout)
        # future.add_done_callback(lambda future: print(future.result()))
        # content, return_code = future.result()
        # if return_code != 0:
        #     raise Exception("ollama is timeout")
        result = [None, None]

        def target():
            result[0], result[1] = self._chat(message, img_path)

        try:
            start_time = time.time()
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(self.remaining_time(start_time + timeout))
            if thread.is_alive():
                raise TimeoutError("ollama is timeout")
        finally:
            thread.join()

        return result[0]


class Agent_ollama_python:

    def __init__(
        self,
        instruction,
        role,
        meta="",
        rag=None,
        model_info=None,
        source=None,
        examplers=None,
        img_path=None,
        logger=None,
        use_cached=False,
        cache=None
    ):
        self.instruction = instruction
        self.role = role
        self.model_info = (
            model_info if model_info is not None else os.environ["MODEL_INFO"]
        ).replace("_", ":")
        # self.source = source if source is not None else os.environ["SOURCE"]
        self.img_path = img_path
        self.logger = logger
        self.content = ""
        self.rag = rag
        self.work_dir = os.environ["work_dir"]
        self.use_cached = use_cached

        self.messages = [
            {"role": "system", "content": json.dumps(self.instruction)},
        ]

        # ollama.show(model_info)

    @time_it
    @retry_on_api_error
    def temp_responses(
        self, message, action, rag=None, img_path=None, timeout=300, debug=False
    ):

        if self.model_info == "gemini-pro":
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses
        else:
            # if self.model_info in ["gpt-3.5", "gpt-4", "gpt-4o", "gpt-4o-mini"]:
            self.messages.append({"role": "user", "content": json.dumps(message)})
            # prompt = """
            #         Rules:
            #         - You MUST select trials exactly as provided (including number and parameters).
            #         - "{context}\n\n"
            #         - You MUST NOT modify, mix, create, or imagine any new parameter combinations.
            #         - You MUST NOT invent new trial numbers.
            #         - Trials already recommended ({used_trial_ids}) MUST NOT be selected again.
            #         "1. ** Analysis of All Recommended Trials **"
            #         "  and summarize performance metrics of recommended trials.\n
            #         "- Evaluate performance trends for hyper-parameters.\n"
            #         "- Highlight promising hyper-parameter combinations."
            #         "2. **Optimization Recommendation**\n"
            #         "- Recommend hyper-parameters for the next {n_jobs} trials from the given search space.\n"
            #         "- Do not output any JSON blocks during the reasoning part."
            #         "- Provide reasoning for each recommendation.\n\n"
            #         "3. **Stop Optimization**\n"
            #         "- Output 'Answer: Yes' ONLY if:\n"
            #         "   * Best result is superior to the baseline.\n"
            #         "   * You're confident further trials won't help.\n"
            #         "- Otherwise, output 'Answer: No with confidence score: <float between 0 and 1>'.\n"
            #         "- Provide a short (1-2 sentences) justification either way.\n"
            #         "Finally, your response should include the following JSON format:\n\n"
            #         ```json
            #         [\n
            #         '  {{"number": ..., "params": {{...}}}},\n'
            #         '  {{"number": ..., "params": {{...}}}},\n'
            #         '  {{"number": ..., "params": {{...}}}}\n'
            #         ]
            #         ```
            #         """

            # prompt = prompt.format(
            #     n_jobs=3,
            #     context=json.dumps(self.messages[1]["content"]["target_trials"]),
            #     used_trial_ids=json.dumps(self.messages[1]["content"]["trials"]),
            # )
            # print_log(prompt, logger=self.logger)
            # # self.logger.info(prompt)
            # self.messages[1] = {"role": "user", "content": prompt}

            print("question: \n", self.messages[1]["content"])
            temperatures = [0.0]

            responses = {}
            for temperature in temperatures:
                if "gpt" in self.model_info:
                    if self.model_info == "gpt-3.5":
                        model_info = "gpt-3.5-turbo"
                    else:
                        model_info = "gpt-4o-mini"
                else:
                    model_info = self.model_info

                response = chat(
                    model=model_info,
                    messages=self.messages,
                    options={"temperature": temperature},
                    stream=True,
                )
            print("response: \n")
            content = ""
            for chunk in response:
                # print(chunk.choices[0].delta.content, end="")
                # delta_content = chunk.choices[0].delta.content
                print(chunk["message"]["content"], end="", flush=True)
                delta_content = chunk["message"]["content"]
                if delta_content is not None:
                    content += delta_content
            print()
            print_log(content, logger=self.logger)
            responses[temperature] = content  # response.choices[0].message.content

            return responses[0.0]


class Agent:

    def __init__(
        self,
        instruction,
        role,
        meta,
        rag=False,
        examplers=None,
        model_info=None,
        img_path=None,
        logger=None,
        cache=None
    ):

        self.source = "ollama"
        self.instruction = instruction
        self.role = role
        self.img_path = img_path
        self.logger = logger
        self.model_info = model_info
        self.meta = meta

        self.cache = cache

        if model_info is None:
            try:
                model_info = os.environ["MODEL_INFO"].split(",")
                self.model_info = str(
                    np.random.choice(model_info, size=1, replace=True)[0]
                )

            except:
                self.model_info = (
                    model_info if model_info is not None else os.environ["MODEL_INFO"]
                )

        if self.model_info == "gemini-pro":
            self.model = genai.GenerativeModel("gemini-pro")
            self._chat = self.model.start_chat(history=[])
        elif "claude" in self.model_info:
            import anthropic
            self.source = "claude"
            self.client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
        elif "deepseek" in self.model_info and "deepseek-r1" not in self.model_info:
            self.source = "openai"
            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )

        elif "gpt" in self.model_info:
            self.source = "openai"
            if rag:
                from langchain_community.chat_models import ChatOpenAI
                from langchain.callbacks.streaming_stdout import (
                    StreamingStdOutCallbackHandler,
                )

                handler = StreamingStdOutCallbackHandler()
                self.client = ChatOpenAI(
                    model_name=self.model_info,
                    temperature=0.7,
                    streaming=True,
                    callbacks=[handler],
                )
            else:
                self.client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
        elif "ollama" == self.source:
            self.source = "ollama"
            if rag:
                from langchain_community.llms import Ollama
                from langchain.callbacks.streaming_stdout import (
                    StreamingStdOutCallbackHandler,
                )

                handler = StreamingStdOutCallbackHandler()
                self.client = Ollama(
                    model=self.model_info, streaming=True, callbacks=[handler]
                )
            else:
                self.client = ollama
                # self.client = OpenAI(
                #     api_key="EMPTY",
                #     base_url="http://localhost:11434/v1/",
                # )
        elif "vllm" == self.source:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1/",
            )
        else:
            raise ValueError(f"{self.model_info} doesn't exist.")

        self.messages = []
        if "deepseek" in self.model_info and "deepseek-r1" not in self.model_info:
            self.source = "deepseek"
            if self.instruction is not None:
                self.messages = [
                    {
                        "role": "system",
                        "content": json.dumps(self.instruction),
                        # "content": f"<think>\n\n{json.dumps(self.instruction)}</think>",
                    },
                ]

        else:
            if self.instruction is not None:
                self.messages = [
                    {"role": "system", "content": json.dumps(self.instruction)},
                ]

        self.vis_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def reset(self):
        self.messages = self.messages[:1]

    # @time_it
    # @retry_on_api_error
    def temp_responses(
        self,
        message,
        action,
        use_cached=False,
        img_path=None,
        temperatures=[0.0],
        debug=False,
        timeout=300,
    ):

        # if rag is not None:
        #     trials = rag["trials"]
        #     used_trial_ids = rag["used_trial_ids"]
        #     rag_chat = self.build_rag(rag["args"], trials, used_trial_ids)
        #     return rag_chat({"query": message})[
        #         "result"
        #     ]  # "Recommend parameter configuration."
        if self.cache is not None:
            content = self.cache.load_saved_messages(
                action, self.role, use_cached
            )

            if self.cache.saved_as_path != self.cache.cache_path:
                self.cache.do_save_messages(action, content, self.role, use_cached)
        else:
            content = None

        cached = content is not None

        if self.model_info == "gemini-pro":
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        else:
            if img_path is None:
                if isinstance(message, list):
                    self.messages.extend(message)
                else:
                    self.messages.append({"role": "user", "content": message})
            else:

                if isinstance(img_path, str):
                    image_base64 = img_path
                else:
                    if isinstance(img_path, np.ndarray):
                        pil_image = Image.fromarray(
                            (img_path / img_path.max() * 255).astype(np.uint8)
                        )
                    else:
                        pil_image = img_path
                    image_base64 = encode_image_to_base64(pil_image)
                if "claude" in self.model_info:
                    message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": message},
                        ],
                    }
                else:
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                self.messages.append(message)

            role_supp = ""
            if img_path is not None:
                role_supp = " - image analyzed by gpt-4.1-mini"
                msg = copy.deepcopy(self.messages)
                msg[1]["content"][0]["text"] = (
                    self.vision_prompt + "\n\n" + self.messages[1]["content"][0]["text"]
                )
                if not cached:
                    response = self.vis_client.chat.completions.create(
                        model="gpt-4.1-mini", messages=msg, stream=True, temperature=0.0
                    )
                    content = ""
                    print(f"## {self.role} ( {self.meta} - {role_supp[2:]}): ", end="")
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            print(str(chunk.choices[0].delta.content), end="")
                            delta_content = chunk.choices[0].delta.content
                            if delta_content is not None:
                                content += delta_content
                    print()
                    print_log(
                        f"## {self.role}  ( {self.meta} - {role_supp[2:]}): {content}",
                        logger=self.logger,
                    )
                # intrgrated_message = prompt + content + "\n\n" + message["content"][0]["text"]
                # self.messages.append{"role": "assistant", "content": intrgrated_message}
                # self.messages[1]["content"] = msg[1]["content"]
                if ("deepseek" in self.model_info or self.source == "ollama"):
                    text = self.messages[1]["content"][0]["text"]
                    self.messages[1]["content"] = self.vision_prompt
                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append({"role": "user", "content": self.vision_prompt_part + "\n\n" + text})
                else:
                    text = self.messages[1]["content"][0]["text"]
                    self.messages[1]["content"] = self.vision_prompt
                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append(
                        {"role": "user", "content": self.vision_prompt_part + "\n\n" + text}
                    )
                if not cached:
                    self.cache.do_save_messages(
                        f"{action}",
                        self.messages[-2]["content"],
                        self.role,
                        use_cached,
                    )

            # reduce token usage, gpt model don't conduct review
            if img_path is not None and "gpt" in self.model_info:
                self.messages.pop()
                print(content)
                return content
            elif "gpt" not in self.model_info and img_path is not None and self.cache is not None:
                content = self.cache.load_saved_messages(
                        f"{action}_review_assessment", self.role, use_cached
                    )

            if content is not None:
                self.messages.append({"role": "assistant", "content": content})
                print(content)
                return content

            # prompt = """
            #         "Only use the existing parameter sets listed below. Do not mix, modify, or create new values.\n\n"
            #         "{context}\n\n"
            #         "The following trials have already been recommended and must not be repeated:\n"
            #         "{used_trial_ids}\n\n"
            #         "1. ** Analysis of All Recommended Trials **"
            #         "  and summarize performance metrics of recommended trials.\n
            #         "- Evaluate performance trends for hyper-parameters.\n"
            #         "- Highlight promising hyper-parameter combinations."
            #         "2. **Optimization Recommendation**\n"
            #         "- Recommend hyper-parameters for the next {n_jobs} trials from the given search space.\n"
            #         "- Do not output any JSON blocks during the reasoning part."
            #         "- Provide reasoning for each recommendation.\n\n"
            #         "3. **Stop Optimization**\n"
            #         "- Output 'Answer: Yes' ONLY if:\n"
            #         "   * Best result is superior to the baseline.\n"
            #         "   * You're confident further trials won't help.\n"
            #         "- Otherwise, output 'Answer: No with confidence score: <float between 0 and 1>'.\n"
            #         "- Provide a short (1-2 sentences) justification either way.\n"
            #         "Finally, your response should include the following JSON format:\n\n"
            #         ```json
            #         [\n
            #         '  {{"number": ..., "params": {{...}}}},\n'
            #         '  {{"number": ..., "params": {{...}}}},\n'
            #         '  {{"number": ..., "params": {{...}}}}\n'
            #         ]
            #         ```
            #         """

            # prompt = prompt.format(
            #     n_jobs=3,
            #     context=json.dumps(self.messages[1]["content"]["target_trials"]),
            #     used_trial_ids=json.dumps(self.messages[1]["content"]["trials"]),
            # )
            # print_log(prompt, logger=self.logger)
            # # self.logger.info(prompt)
            # self.messages[1] = {"role": "user", "content": prompt}

            print("response: \n")
            responses = {}
            for temperature in temperatures:
                print("#####")
                print(f"## {self.role} ({self.meta} - {self.model_info}{role_supp}): ", end="")
                if "claude" in self.model_info:
                    system_prompt = self.messages[0]["content"]
                    user_prompt = self.messages[1:]
                    with self.client.messages.stream(
                        system=system_prompt,
                        max_tokens=256,
                        messages=user_prompt,
                        model=self.model_info,
                        temperature=(
                            temperature if temperature != "default" else anthropic.NotGiven
                        ),
                    ) as stream:
                        content = ""
                        for text in stream.text_stream:
                            print(text, end="", flush=True)
                            if text is not None:
                                content += text
                    print()
                    self.messages.append({"role": "assistant", "content": content})
                    print_log(
                        f"## {self.role} ({self.model_info}): {content}",
                        logger=self.logger,
                    )
                    responses[temperature] = content
                elif self.source == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_info,
                        messages=self.messages,
                        temperature=(
                            temperature
                            if temperature != "default"
                            else openai.NotGiven
                        ),
                        stream=True,
                    )
                    content = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            print(str(chunk.choices[0].delta.content), end="")
                            delta_content = chunk.choices[0].delta.content
                            if delta_content is not None:
                                content += delta_content
                    print()
                    self.messages.append({"role": "assistant", "content": content})
                    print_log(
                        f"## {self.role} ({self.model_info} ): {content}",
                        logger=self.logger,
                    )
                    responses[temperature] = content

                elif self.source == "ollama":
                    response = chat(
                        model=self.model_info,
                        messages=self.messages,
                        options=(
                            {"temperature": temperature}
                            if temperature != "default"
                            else None
                        ),
                        stream=True,
                    )
                    content = ""
                    for chunk in response:
                        # print(chunk.choices[0].delta.content, end="")
                        # delta_content = chunk.choices[0].delta.content
                        print(chunk["message"]["content"], end="", flush=True)
                        delta_content = chunk["message"]["content"]
                        if delta_content is not None:
                            content += delta_content
                    print()
                    self.messages.append({"role": "assistant", "content": content})
                    print_log(
                        f"## {self.role} ({self.model_info} ): {content}",
                        logger=self.logger,
                    )
                    responses[temperature] = content

                elif "deepseek" in self.model_info:
                    prompt = self.messages.pop()["content"]
                    self.messages.append({"role": "user",
                        "content": f"<think>\n\n{prompt + "\n\n" + self.messages[0]["content"]}</think>"})

                    reasoning_content = "<think> "
                    content = ""
                    response = self.client.chat.completions.create(
                        model=self.model_info,
                        messages=self.messages,
                        stream=True,
                        temperature=(
                            temperature if temperature != "default" else openai.NotGiven
                        ),
                        timeout=httpx.Timeout(10.0, read=20.0),
                        # top_p=0,
                        # top_p=0.95,
                        # max_tokens=2048,
                    )
                    print("<think> ", end="")
                    for chunk in response:
                        delta_reasoning_content = chunk.choices[
                            0
                        ].delta.reasoning_content
                        if delta_reasoning_content is not None:
                            print(delta_reasoning_content, end="")
                            reasoning_content += delta_reasoning_content
                        elif reasoning_content != "<think> " and content == "":
                            print(" <\\think>")

                        delta_content = chunk.choices[0].delta.content
                        if delta_content is not None:
                            print(delta_content, end="")
                            content += delta_content

                    if "answer:" not in content.lower():
                        content = "Answer:"+content
                    self.messages.append({"role":"assistant", "content": content})
                    print_log(f"## {self.role} ({self.model_info} ): {reasoning_content}", logger=self.logger)
                    print_log(f"## {self.role} ({self.model_info} ): {content}", logger=self.logger)
                    responses[temperature] = content

            if "deepseek" in self.model_info and self.cache is not None and "deepseek-r1" not in self.model_info:
                self.cache.do_save_messages(
                        f"think-{action}",
                        reasoning_content,
                        self.role,
                        use_cached,
                    )
                if img_path is not None:
                    self.cache.do_save_messages(
                        f"{action}_review_assessment",
                        content,
                        self.role,
                        use_cached,
                    )
                else:
                    self.cache.do_save_messages(
                            f"{action}",
                            content,
                            self.role,
                            use_cached,
                        )
            elif self.cache is not None:
                self.cache.do_save_messages(
                    f"{action}",
                    content,
                    self.role,
                    use_cached,
                )

            return responses[0.0]
