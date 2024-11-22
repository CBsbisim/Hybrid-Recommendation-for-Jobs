# -*- coding: utf-8 -*-
# @Time    : 2024-11-21
# @Author  : jinzhuang，chenximin
 
from flask_apscheduler import APScheduler
from main import get_data_for_userid, save_data_by_cycle
from flask import Flask, current_app, request
from logger import *
import traceback

# 设置时区
import os
# 设置系统环境变量的timezone 
os.environ['TZ']= 'Asia/Shanghai'
from apscheduler.schedulers.background import BackgroundScheduler
# 初始化app flask对象
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 设置APScheduler注册的timezone
sched = APScheduler(BackgroundScheduler(timezone="Asia/Shanghai"))


# 自动保存模型--定时任务
def save_data():
    save_data_by_cycle()

 
@app.route('/recommend', methods=['get', 'post'])
def recommend():
    result = {}
    try:
        user_id = request.args.get('user_id')
        # res = recommend_by_id_csv(int(user_id))
        result['code'] = 200
        result['data'] = get_data_for_userid(user_id)
    except Exception as e:
        result['code'] = 404
        result['data'] = "出错啦"
        log.error(traceback.format_exc())
    return result

# app_ctx = app.app_context()  
# app.app_context()   # 是一个上下文表达式，它返回了一个上下文管理器AppContext() 
# app_ctx.push()
# print(current_app.name)  # 要执行的代码
# log.info(current_app.name)
# app_ctx.pop()
 
 
if __name__ == "__main__":
    # 定时任务 
    sched.add_job(func=save_data, id='1', trigger='cron', hour='4')
    sched.init_app(app=app)
    sched.start()
    # app.run(debug=False, host='0.0.0.0', port=10002)
    from gevent import pywsgi
    server = pywsgi.WSGIServer(('0.0.0.0', 10002), app, log=None)
    server.serve_forever()
    

   # //分服务器计算

