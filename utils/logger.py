# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)#获得logg对象
    logger.setLevel(logging.DEBUG) #日志会记录设置级别以上的日志 NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
   
    # don't log results for the non-master process，addHandler方法添加0到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示
    #handler将日志记录（log record）发送到合适的目的地（destination），
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout) #建立一个streamhandler来把日志打在CMD窗口上 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s") #定义了最终log信息的顺序,结构和内容
    ch.setFormatter(formatter)
    logger.addHandler(ch) #将相应的handler添加在logger对象中

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w') #保存的位置
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter) 
        logger.addHandler(fh)

    return logger #获得的logger对象有两个handler一个用于输出在命令行中，一个用于保存在目的目录的txt中
