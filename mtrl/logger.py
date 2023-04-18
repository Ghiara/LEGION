# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
import json
import os
import time
from functools import singledispatch
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import datetime


@singledispatch
def serialize_log(val):
    """Used by default."""
    return val


@serialize_log.register(np.float32)
def np_float32(val):
    return np.float64(val)


@serialize_log.register(np.int64)
def np_int64(val):
    return int(val)


class Meter(object):
    def __init__(self):
        pass

    def update(self, value, n=1):
        pass

    def value(self):
        pass


class AverageMeter(Meter):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class CurrentMeter(Meter):
    def __init__(self):
        pass

    def update(self, value, n=1):
        self._value = value

    def value(self):
        return self._value


class MetersGroup(object):
    def __init__(self, file_name, formating, mode: str, retain_logs: bool):
        self._file_name = file_name
        self._mode = mode
        if not retain_logs:
            if os.path.exists(file_name):
                os.remove(file_name)
        self._formating = formating
        self._meters: Dict[str, Meter] = {}

    def log(self, key, value, n=1):
        if key not in self._meters:
            metric_type = self._formating[key][2]
            if metric_type == "average":
                self._meters[key] = AverageMeter()
            elif metric_type == "constant":
                self._meters[key] = CurrentMeter()
            else:
                raise ValueError(f"{metric_type} is not supported by logger.")
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            data[key] = meter.value()
        data["mode"] = self._mode
        return data

    def _dump_to_file(self, data):
        data["logbook_timestamp"] = time.strftime("%I:%M:%S%p %Z %b %d, %Y")
        with open(self._file_name, "a") as f:
            f.write(json.dumps(data, default=serialize_log) + "\n")

    def _format(self, key, value, ty):
        template = "%s: "
        if ty == "int":
            template += "%d"
        elif ty == "float":
            template += "%.04f"
        elif ty == "time":
            template += "%.01f s"
        elif ty == "str":
            template += "%s"
        else:
            raise "invalid format type: %s" % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = ["{:5}".format(prefix)]
        for key, (disp_key, ty, _) in self._formating.items():
            if key in data:
                value = data.get(key, 0)
                if disp_key is not None:
                    pieces.append(self._format(disp_key, value, ty))
        print("| %s" % (" | ".join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["step"] = step
        self._dump_to_file(data) # wirte the logging data to files
        self._dump_to_console(data, prefix) # print logging data to console
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, config, retain_logs: bool = False):
        self._log_dir = log_dir
        self.config = config
        
        ##############################
        # use tensorboard as logging #
        ##############################
        self.use_tb = self.config.logger.use_tb
        if self.use_tb:
            self.tb_writer = SummaryWriter(self._log_dir+'/tb_logger_{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M")))
        else:
            self.tb_writer = None
        
        if "metaworld" in self.config.env.name:
            num_envs = int(
                "".join(
                    [
                        x
                        for x in self.config.env.benchmark._target_.split(".")[1]
                        if x.isdigit()
                    ]
                )
            )
        else:
            env_list: List[str] = []
            for key in self.config.metrics:
                if "_" in key:
                    mode, submode = key.split("_")
                    # todo: should we instead throw an error here?
                    if mode in self.config.env and submode in self.config.env[mode]:
                        env_list += self.config.env[mode][submode]
                else:
                    if key in self.config.env:
                        env_list += self.config.env[key]
            num_envs = len(set(env_list))

        def _get_formatting(
            current_formatting: List[List[str]],
        ) -> Dict[str, List[str]]:
            formating: Dict[str, List[str]] = {
                _format[0]: _format[1:] for _format in current_formatting
            }
            if num_envs > 0:
                keys = list(formating.keys())
                for key in keys:
                    if key.endswith("_"):
                        value = formating.pop(key)
                        for index in range(num_envs):
                            new_key = key + str(index)
                            if value[0] is None:
                                abbr = None
                            else:
                                abbr = value[0] + str(index)
                            formating[new_key] = [abbr, *value[1:]]
            return formating

        self.mgs = {
            key: MetersGroup(
                os.path.join(log_dir, f"{key}.log"),
                formating=_get_formatting(current_formatting=value),
                mode=key,
                retain_logs=retain_logs,
            )
            for key, value in self.config.metrics.items()
        }

    def log(self, key, value, step, n=1, tb_log:bool=True, only_tb_log:bool=False):
        
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        mode, key = key.split("/", 1)
        
        if not only_tb_log:
            self.mgs[mode].log(key, value, n)

        #################################
        # using tensorboard as notation #
        #################################
        if self.use_tb and tb_log:
            self.tb_writer.add_scalar(tag='{}/{}'.format(mode, key), scalar_value=value, global_step=step)

    def log_text(self, keys, content, step):
        '''
        used to log important hyperparameters like env idx to embedding map
        '''
        assert keys.startswith("train") or keys.startswith("eval")
        mode, key = keys.split("/", 1)
        if self.tb_writer is None:
            with open(self._log_dir+'/{}_{}.json'.format(mode, key), 'w') as fp:
                json.dump(content, fp)
        else:
            if type(content) == dict:
                count = 0
                for k, val in content.items():
                    self.tb_writer.add_text(tag='{}/{}'.format(mode,key), text_string='{}:{}'.format(k, val), global_step=count)
                    count += 1
            else:
                self.tb_writer.add_text(tag='{}/{}'.format(mode,key), text_string=str(content), global_step=step)


    def dump(self, step):
        for key in self.mgs:
            self.mgs[key].dump(step, key) # dump to file & console



##########################################################
# TODO: set up continuous learning metrics manager #######
##########################################################
class CRL_Metrics():
    def __init__(self, save_dir) -> None:

        self.save_dir = save_dir

        self.subtask_success = []
        self.subtask_reward = []
        self.success = []
        self.reward = []

    
    def add(self, reward, success_rate):
        # add data of single subtask
        self.subtask_reward.append(reward)
        self.subtask_success.append(success_rate)

    def update(self):
        # update total metrics
        # reset subtask metrics
        self.success.append(self.subtask_success)
        self.reward.append(self.subtask_reward)
        self.subtask_success = []
        self.subtask_reward = []


    def save_metrics(self):
        metrics = dict(
            success = self.success,
            reward = self.reward
        )
        np.savez(self.save_dir+'/metrics.npz', **metrics)
    
    def to_csv(self):
        metrics = np.array(self.success).T # axis 0 eval success rate after every queue phase, axis 1 queue of env
        metrics = pd.DataFrame(metrics, columns=['reach-v1','push-v1',
                                                 'pick-place-v1','door-open-v1', 
                                                 'drawer-open-v1','drawer-close-v1', 
                                                 'button-press-topdpwn-v1','peg-insert-side-v1',
                                                 'window-open-v1','window-close-v1'])
        metrics.to_csv(self.save_dir+'/eval_success.csv')

