# -*- coding: utf-8 -*-
import logging
import time
import sys
import os
from logging import handlers
'''
日志logger类
'''
 
# 创建logs文件夹
cur_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(cur_path, 'error_logs')
# 如果不存在这个logs文件夹，就自动创建一个
if not os.path.exists(log_path): os.mkdir(log_path)
 
 
class Log(object):
    def __init__(self):
        # 文件的命名
        self.logname = os.path.join(log_path, '%s.log' % time.strftime('%Y_%m_%d'))
        logging.basicConfig()
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.ERROR)
        self.logger.propagate = False
        # 日志输出格式
        self.formatter = logging.Formatter('[%(asctime)s] - %(filename)s] - %(levelname)s: %(message)s')
 
    def __console(self, level, message):
        # # 创建一个FileHandler，用于写到本地
        # fh = logging.FileHandler(self.logname, 'a', encoding='utf-8')
        # fh.setLevel(logging.INFO)
        # fh.setFormatter(self.formatter)
        # self.logger.addHandler(fh)

        # # 创建一个StreamHandler,用于输出到控制台
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.ERROR)
        # ch.setFormatter(self.formatter)
        # self.logger.addHandler(ch)

        # 日志文件按天进行保存，每天一个日志文件
        file_handler = handlers.TimedRotatingFileHandler(filename=self.logname, when='D', backupCount=1, encoding='utf-8')
        # 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
        # file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=1*1024*1024*1024, backupCount=1, encoding='utf-8')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        # 这两行代码是为了避免日志输出重复问题
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # # 关闭打开的文件
        # fh.close()
        # ch.close()
 
    def debug(self, message):
        self.__console('debug', message)
 
    def info(self, message):
        self.__console('info', message)
 
    def warning(self, message):
        self.__console('warning', message)
 
    def error(self, message):
        self.__console('error', message)
 
 
log = Log()
 
if __name__ == '__main__':
    pass